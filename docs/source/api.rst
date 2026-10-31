API Reference
=============

This page documents the public API of QuSpect.

Top-Level
---------

.. automodule:: quspect
   :members:
   :imported-members:

Circuit
-------

.. automodule:: quspect.circuit
   :members:

.. automodule:: quspect.circuit.circuit
   :members:
   :special-members: __init__, __len__, __iter__, __getitem__

.. automodule:: quspect.circuit.instruction
   :members:

.. automodule:: quspect.circuit.parameter
   :members:

Algorithms
----------

.. automodule:: quspect.algorithms
   :members:

.. automodule:: quspect.algorithms.qft
   :members:

Gates
-----

.. automodule:: quspect.gates
   :members:

.. automodule:: quspect.gates.gate
   :members:

.. automodule:: quspect.gates.standard
   :members:
   :exclude-members: StandardGate  (documented via Gate)

.. automodule:: quspect.gates.parameterized
   :members:
   :exclude-members: ParameterizedGate  (documented via Gate)

Single-Qubit Synthesis
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.gates.synthesis
   :members:

Matrix Utilities
~~~~~~~~~~~~~~~~

.. automodule:: quspect.gates._matrix_utils
   :members:
   :undoc-members:

Native Backend
--------------

.. automodule:: quspect.native
   :members:

Pulse Envelopes
---------------

.. automodule:: quspect.pulse
   :members:
   :imported-members:
   :undoc-members:

Variational Objectives
----------------------

.. automodule:: quspect.variational
   :members:

Quadratic Optimization
----------------------

.. automodule:: quspect.optimization
   :members:

.. automodule:: quspect.optimization.expression
   :members:

.. automodule:: quspect.optimization.program
   :members:

.. automodule:: quspect.optimization.compiler
   :members:

.. automodule:: quspect.optimization.model
   :members:

.. automodule:: quspect.optimization.conversion
   :members:

.. automodule:: quspect.optimization.hardware
   :members:

.. automodule:: quspect.optimization.codec
   :members:

.. automodule:: quspect.optimization.native_solver
   :members:

.. automodule:: quspect.optimization.isingq
   :members:

DAG
---

.. automodule:: quspect.dag
   :members:

.. automodule:: quspect.dag.dag_node
   :members:

.. automodule:: quspect.dag.dag_circuit
   :members:

.. automodule:: quspect.dag.compiler
   :members:

Synthesis
---------

.. automodule:: quspect.synthesis
   :members:

.. automodule:: quspect.synthesis.two_qubit
   :members:

Transpiler: Block Consolidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._consolidate
   :members:

Transpiler: Target Basis
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._basis
   :members:

Transpiler: Single-Qubit Minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._gauge
   :members:

Transpiler: SWAP-Face Blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._swap_face
   :members:

Transpiler: Depth-Oriented Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._schedule
   :members: schedule_for_depth

Transpiler: One-Call Transpile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._transpile
   :members: transpile, TranspileResult, TranspileStats

Transpiler: SABRE Routing Candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: quspect.transpiler._mapping.RoutingCandidate
   :members:
   :no-index:

.. automethod:: quspect.transpiler._mapping.SABRE.route_candidates
   :no-index:

.. autofunction:: quspect.transpiler._mapping.estimate_two_qubit_cost
   :no-index:

Transpiler: Low-Dilation Layouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._embedding
   :members: dilation_layouts

Transpiler: Exact Window Re-Routing (LNS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._route_lns
   :members: exact_route, refine_routing, refine_plan, deepen_plan,
      routing_cost, node_cost_model, node_weights, lns_available,
      ExactRouting, RefinedRouting

Transpiler: Routing of SU(4)-Block Circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._layer_route
   :members: layer_route_plan, layer_route_events, su4_route_events, block_layers,
      path_order, cycle_order

Transpiler: SABRE Routing Core
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._routing
   :members: route_dag, routing_backend, cost3, incident_edges, ROUTING_BACKEND_ENV

Transpiler: High-Level Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._highlevel
   :members:

Transpiler: Phase Folding
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._phase_fold
   :members:
   :exclude-members: rotations_in, rotations_out

Transpiler: Three-Qubit Block Resynthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._three_qubit
   :members: ThreeQubitBlockPass, ThreeQubitBlockStats, synthesize_three_qubit
   :exclude-members: blocks_examined, blocks_replaced, two_qubit_before,
      two_qubit_after

Transpiler: Toffoli Routing and Synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: quspect.transpiler._toffoli
   :members: ccx_candidates, ccx_path_decomposition, ccx_triangle_decomposition,
      synthesize_ccx_for_coupling
