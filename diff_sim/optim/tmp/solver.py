
####################################
# Initial solve function
####################################
# def solve(m: Model, d: Data) -> Data:
#   """Finds forces that satisfy constraints using conjugate gradient descent."""

#   def cond(ctx: _Context) -> jax.Array:
#     improvement = _rescale(m, ctx.prev_cost - ctx.cost)
#     gradient = _rescale(m, math.norm(ctx.grad))

#     done = ctx.solver_niter >= m.opt.iterations
#     done |= improvement < m.opt.tolerance
#     done |= gradient < m.opt.tolerance

#     return ~done

#   def body(ctx: _Context) -> _Context:
#     ctx = _linesearch(m, d, ctx)
#     prev_grad, prev_Mgrad = ctx.grad, ctx.Mgrad  # pylint: disable=invalid-name
#     ctx = _update_constraint(m, d, ctx)
#     ctx = _update_gradient(m, d, ctx)

#     if m.opt.solver == SolverType.NEWTON:
#       search = -ctx.Mgrad
#     else:
#       # polak-ribiere:
#       beta = jp.dot(ctx.grad, ctx.Mgrad - prev_Mgrad)
#       beta = beta / jp.maximum(mujoco.mjMINVAL, jp.dot(prev_grad, prev_Mgrad))
#       beta = jp.maximum(0, beta)
#       search = -ctx.Mgrad + beta * ctx.search
#     ctx = ctx.replace(search=search, solver_niter=ctx.solver_niter + 1)

#     return ctx

#   # warmstart:
#   qacc = d.qacc_smooth
#   if not m.opt.disableflags & DisableBit.WARMSTART:
#     warm = _Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=False)
#     smth = _Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
#     qacc = jp.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
#   d = d.replace(qacc=qacc)

#   ctx = _Context.create(m, d)
#   if m.opt.iterations == 1:
#     ctx = body(ctx)
#   else:
#     ctx = jax.lax.while_loop(cond, body, ctx)

#   d = d.replace(
#       qacc_warmstart=ctx.qacc,
#       qacc=ctx.qacc,
#       qfrc_constraint=ctx.qfrc_constraint,
#       efc_force=ctx.efc_force,
#   )

#   return d


################################################
# Solve function with Implicit Differentiation
################################################

# The objective function: given qacc (in pytree form) it returns a (structured)
# residual that should be zero at the solution.
def optimality_function(qacc_guess: jp.array, m: Model, d: Data) -> jp.array:
    # Build a "tracked_data" or "ctx" that depends on qacc_guess
    tracked_data = d.replace(qacc=qacc_guess)
    ctx = _Context.create(m, tracked_data, grad=False)
    ctx = _update_gradient(m, d, ctx)
    ctx = ctx.replace(search=-ctx.Mgrad)  # start with preconditioned gradient
    return ctx.grad

def solve_iterative(
    f, 
    qacc_guess: jp.array, 
    m: Model, 
    d: Data
):
    """
    Iterative solver that uses linesearch, etc., in a pure, loop-based fashion.
    All references to `m` and `d` are read-only for shapes/constants.
    """

    def cond_fn(ctx: _Context):
        improvement = _rescale(m, ctx.prev_cost - ctx.cost)
        gradient = _rescale(m, jp.linalg.norm(ctx.grad))
        # Don’t invert the logic with ~(...). Just write it explicitly:
        return (
            (ctx.solver_niter < m.opt.iterations) &
            (improvement > m.opt.tolerance) &
            (gradient > m.opt.tolerance)
        )

    def body_fn(ctx: _Context):
        # linesearch, constraint update, gradient update, etc.
        ctx = _linesearch(m, d, ctx)
        ctx = _update_constraint(m, d, ctx)
        ctx = _update_gradient(m, d, ctx)

        if m.opt.solver == SolverType.NEWTON:
            search = -ctx.Mgrad
        else:
            search = None  # or some default fallback

        return ctx.replace(
            search=search,
            solver_niter=ctx.solver_niter + 1
        )

    # Initialize loop state
    ctx0 = _Context.create(m, d).replace(qacc=qacc_guess)

    # Single while_loop
    final_ctx = jax.lax.while_loop(cond_fn, body_fn, ctx0)

    return final_ctx.qacc, (final_ctx.qfrc_constraint, final_ctx.efc_force)


def solve_iterative(f, qacc_guess, m: Model, d: Data):
  def cond(ctx: _Context):
    improvement = _rescale(m, ctx.prev_cost - ctx.cost)
    gradient = _rescale(m, jp.linalg.norm(ctx.grad))
    # return ~(ctx.solver_niter >= m.opt.iterations
    #           | (improvement < m.opt.tolerance)
    #           | (gradient < m.opt.tolerance))
    return (
            (ctx.solver_niter < m.opt.iterations) &
            (improvement > m.opt.tolerance) &
            (gradient > m.opt.tolerance)
        )

  def body(ctx: _Context):
    ctx = _linesearch(m, d, ctx)
    ctx = _update_constraint(m, d, ctx)
    ctx = _update_gradient(m, d, ctx)
    search = -ctx.Mgrad if m.opt.solver == SolverType.NEWTON else None
    return ctx.replace(search=search, solver_niter=ctx.solver_niter + 1)

  ctx_iter = _Context.create(m, d).replace(qacc=qacc_guess)
  if m.opt.iterations > 1:
    ctx_iter = jax.lax.while_loop(cond, body, ctx_iter)
  else:
    ctx_iter = body(ctx_iter)
  return ctx_iter.qacc, (ctx_iter.qfrc_constraint, ctx_iter.efc_force)

def tangent_solve(g, y):
  J = jax.jacobian(g)(y)  # shape (n, n)
  Q, R = jax.tree.map(lambda x: jp.linalg.qr(x, mode='reduced'), J)
  # use pinv
  # x = jax.tree.map(lambda q, r: jp.linalg.pinv(r) @ q.T @ y, Q, R)
  # use linear solve
  # A = jax.tree.map(lambda J: J.T @ J, J)
  # b = jax.tree.map(lambda J, y: J.T @ y, J, y)
  # x = jax.tree.map(lambda A, b: jp.linalg.solve(A, b), A, b)
  # use qr
  x = jax.tree.map(lambda q, r: jp.linalg.solve(r, q.T @ y), Q, R)
  return x
  

def solve(m: Model, d: Data) -> Data:
  from jax.flatten_util import ravel_pytree
  """
  Solves for qacc using the implicit function theorem-based solver in JAX,
  using jax.lax.custom_root on a flattened version of the root function.

  Parameters:
      m (Model): MuJoCo model instance.
      d (Data): MuJoCo data instance.

  Returns:
      Data: Updated MuJoCo data with solved values.
  """
  f = lambda qacc: optimality_function(qacc, m, d)
  solve_it = lambda f,qacc: solve_iterative(f,qacc, m, d)

  # Optionally use a warmstart if enabled.
  if not (m.opt.disableflags & DisableBit.WARMSTART):
    warm = _Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=True)
    smth = _Context.create(m, d.replace(qacc=d.qacc_smooth), grad=True)
    qacc_guess = jp.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
    # flat_qacc_guess, unravel_fn = ravel_pytree(qacc_guess)
  d = d.replace(qacc=qacc_guess)

  # Run the flat custom_root solver.
  qacc_star, (qfrc_constraint, efc_force) = jax.lax.custom_root(
    f=f,
    initial_guess=qacc_guess,
    solve=solve_it,
    tangent_solve=tangent_solve,
    has_aux=True
  ) # Q, R each shape ~ (n, n)

  # Update the MuJoCo data object with the solved values.
  d = d.replace(
    qacc_warmstart=qacc_star,
    qacc=qacc_star,
    qfrc_constraint=qfrc_constraint,
    efc_force=efc_force
  )

  return d


# from jaxopt import implicit_diff
# from jaxopt import linear_solve
 

#  # 1) Define the Optimality Function (Residual)
# def optimality_function(qacc_guess: jp.array, m: Model, d: Data) -> jp.array:
#     # Temporarily store qacc_guess in d
#     tracked_d = d.replace(qacc=qacc_guess)
#     # Create context with no iteration, just compute constraint + gradient
#     ctx = _Context.create(m, tracked_d, grad=False)
#     ctx = _update_constraint(m, tracked_d, ctx)
#     ctx = _update_gradient(m, tracked_d, ctx)
#     return ctx.grad  # shape = (m.nv,)

# # 2) Define the Iterative Solver (Newton's method)
# @implicit_diff.custom_root(optimality_fun=optimality_function, has_aux=True)
# def iterative_solver(qacc_guess: jax.Array, m: Model, d: Data):
#     """
#     Uses Newton's method to iteratively solve for qacc.
#     """

#     def cond(ctx: _Context) -> jax.Array:
#         improvement = _rescale(m, ctx.prev_cost - ctx.cost)
#         gradient = _rescale(m, jp.linalg.norm(ctx.grad))
#         return ~(ctx.solver_niter >= m.opt.iterations
#                   | (improvement < m.opt.tolerance)
#                   | (gradient < m.opt.tolerance))

#     def body(ctx: _Context) -> _Context:
#         ctx = _linesearch(m, d, ctx)
#         ctx = _update_constraint(m, d, ctx)
#         ctx = _update_gradient(m, d, ctx)
#         search = -ctx.Mgrad if m.opt.solver == SolverType.NEWTON else None
#         return ctx.replace(search=search, solver_niter=ctx.solver_niter + 1)

#     ctx_iter = _Context.create(m, d).replace(qacc=qacc_guess)
#     ctx_iter = jax.lax.while_loop(cond, body, ctx_iter) if m.opt.iterations > 1 else body(ctx_iter)

#     return ctx_iter.qacc, (ctx_iter.qfrc_constraint, ctx_iter.efc_force)  # `has_aux=True`
 


# """
# Solver with Implicit Differentiation using jaxopt.implicit_diff 
# """
# def solve(m: Model, d: Data) -> Data:   
#     # 3) Warmstart qacc
#     qacc = d.qacc_smooth
#     if not m.opt.disableflags & DisableBit.WARMSTART:
#         warm = _Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=False)
#         smth = _Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
#         qacc = jp.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
#     d = d.replace(qacc=qacc)
 
#     # 4) Get Optimal Values from Context
#     qacc, (qfrc_constraint, efc_force) = iterative_solver(qacc, m, d)  # Correct call
 
#     # 5) Update the MuJoCo data object with the solved values
#     d = d.replace(qacc_warmstart=qacc, qacc=qacc, qfrc_constraint=qfrc_constraint, efc_force=efc_force)
#     return d