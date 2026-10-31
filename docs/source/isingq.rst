iSing cloud optimization
========================

Install ``quspect[isingq]`` to submit optimization models to iSing. This backend
accepts binary quadratic models and logical Ising problems; it does not execute
quantum circuits. The native extension is not required for compilation or cloud
transport.

Compile without submitting
--------------------------

.. code-block:: python

   from quspect.optimization import bqm_from_qubo_matrix, compile_isingq

   model = bqm_from_qubo_matrix([[-1, 2], [0, -3]], offset=5)
   program = compile_isingq(model)
   print(program.sha256)
   print(program.decode([0, 1])["objective_value"])  # 2, including the offset

The uploaded CSV is upper triangular and minimizes ``x.T @ Q @ x``. Conversion
preserves the original variable order and constants. Logical spins use
``s = 2x - 1``; maximization negates the uploaded coefficients. Decoding evaluates
the original objective independently of the provider's reported energy. Do not
upload ``HardwareIsingProgram.matrix`` unchanged: its ``-s.T @ M @ s`` convention
is different.

For a symbolic ``QuadraticProgram``, first call ``compile_program`` and pass its
``model`` to ``compile_isingq``. Keep that compilation result to call
``is_feasible`` and ``evaluate_objective`` on returned bits. The uploaded model
includes penalties; the cloud result alone cannot establish constraint feasibility
or global optimality.

Submit and recover
------------------

The following submit call creates a real hardware task and may consume quota or
credit. Supply the key explicitly from your credential store; never save it in a
job reference.

.. code-block:: python

   import os
   from quspect.optimization import IsingQClient

   with IsingQClient(os.environ["ISINGQ_API_KEY"]) as client:
       task_id = client.submit(program, calculate_count=1, use_credit=False)
       # Persist task_id and the original model before polling.
       snapshot = client.poll(task_id, program)

Polling is a separate read and can be resumed in a new client. ``calculate_count``
is the vendor's calculation count, not circuit shots or a promise of that many
returned samples. The API returns a best vector, not a frequency distribution.
The SDK does not automatically retry submission. An ``IsingQError`` with
``submission_outcome == "unknown"`` requires reconciliation with the provider
before starting another run. Failures before creation use ``not_created``.
Polling failures use ``not_applicable``: retry the read using the saved task ID,
never treat a read failure as evidence that submission failed.

Cloud limitations
-----------------

The adapter follows the published ``plectrum-core`` 0.1.2 and ``isingq-toolkit``
0.5.0 protocol. It uses bounded HTTPS requests and sends only temporary OSS form
credentials to the upload host. API account credentials remain on the API origin.
There is no public device discovery or cancellation contract. ``computer_type_id``
is a vendor routing value, not proof of which physical machine executes a job.
``gear`` is not exposed because the published cloud implementation does not send
it. Coefficient quantization and its admissible ranges remain provider-defined;
this adapter does not silently round coefficients or claim exact hardware energy
equivalence.

The conversion and transport are covered by exhaustive small-model identities and
mocked protocol tests. Live account acceptance is a separate deployment check.

References: `iSing SDK documentation <https://help-center.isingq.com/8744708m0>`_,
`Plectrum source <https://github.com/ising-tech/plectrum-core>`_, and
`iSing MCP toolkit <https://github.com/ising-tech/isingq-toolkit>`_.
