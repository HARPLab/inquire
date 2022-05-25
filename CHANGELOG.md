5/10/22:

- Pat:

  - Change: utils/learning.py:gradient(): Learning rate 0.005 -> 0.0025.\
    Reason: Pizza making domain wouldn't converge; LDS sometimes wouldn't converge.\
    Effect: So far so good w.r.t. convergence.

5/04/22:

- Pat:

  - Change: utils/learning.py:gradient(): Learning rate 0.001 -> 0.005.\
    Reason: See if convergence time decreases.\
    Effect: TBD for Lunar Lander; faster convergence for linear SoE.

4/07/22:

- Pat:

  - Change: Removed maxiters and maxfun from BFGS optimization.\
    Reason: See if resulting optimizations are superior.\
    Effect: TBD for Lunar Lander; faster optimization for linear SoE.

- Pat:

  - Change: Stubbed the inner for-loop in evaluation.py.\
    Reason: Program took too long to execute.\
    Effect: Easier debugging.

3/24/22:

- Pat:

  - Change: Changed indexing of perf\_mat in evaluation.py.\
    Reason: Indexing errors keep cropping up for longer trials.\
    Effect: Small test ran to completion. TBD w.r.t. long tests.

3/22/22:

- Pat:

  - Change: utils/learning.py:gradient(): Learning rate 0.01 -> 0.001.\
    Reason: GD again stopped converging when the feedback matrix grew.\
    Effect: GD converges

3/21/22:

- Pat:

  - Change: utils/learning.py:gradient(): Learning rate 0.05 -> 0.01.\
    Reason: GD wasn't converging when various interaction types were present in
            the feedback matrix.\
    Effect: GD converges

3/18/22:

- Pat:

  - Change: Reverted feature-function change.\
    Reason: Realized it's needed for rewarding similarity.
  - Change: Normalized weights by L2 norm (in environment's gen. rand. rwd()).\
    Reason: See if it lands optimally.
    Effect: Optimization takes less time on average. Trajectories improved!

3/15/22:

- Pat:

  - Change: Removed exponentiation from lunar lander's feature function and put
            it in optimal\_trajectory\* reward definition.\
    Reason: If Inquire wants raw features, exponentiation messes that up. To optimize
            for the weights, however, we need to exponentiate the negative feature
            values to properly incorporate them into the minimization process.

  - Change: Added "who\_called\_me" argument to \*\_optimal\_trajectory\_\* functions.\
    Reason: Track who/what calls this function to see the corresponding trajectories.

  - Change: Stubbed the equality assertion on reward in evaluation.py.\
    Reason: Let the process run to observe agent's behavior when choosing
            interaction types.
