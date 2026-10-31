Quickstart
==========

Installation
------------

.. code-block:: bash

   python -m pip install quspect

Binary wheels with the C++ simulation engine are available for Linux x86_64,
Linux aarch64, Windows x86_64 and macOS arm64 (Python 3.10 -- 3.15).  See
:doc:`installation` for supported platforms, building from source and
development installs.

Basic Usage
-----------

Build a Bell state circuit:

.. code-block:: python

   from quspect import QuantumCircuit

   circ = QuantumCircuit(name="bell")
   circ.h(0)          # Hadamard on qubit 0
   circ.cx(0, 1)      # CNOT: control=0, target=1
   circ.measure([0, 1], [0, 1])

   print(circ.num_qubits)   # 2 — dynamically inferred
   print(circ.num_clbits)   # 2
   print(len(circ))         # 3 instructions

   # View circuit diagram
   print(circ.draw())

   # Or render as SVG in a notebook / saveable string wrapper
   svg = circ.svg()
   print(str(svg)[:80])

Parameterized Circuits
----------------------

.. code-block:: python

   from quspect import QuantumCircuit, Parameter

   theta = Parameter("θ")
   phi = Parameter("φ")

   circ = QuantumCircuit(name="variational")
   circ.rx(theta, 0)
   circ.ry(phi, 1)
   circ.cx(0, 1)

   print(circ.parameters)  # {Parameter('θ'), Parameter('φ')}

   # Bind parameters to concrete values
   bound = circ.bind_parameters({theta: 0.5, phi: 1.2})

Circuit Composition
--------------------

.. code-block:: python

   # Build a reusable sub-circuit
   sub = QuantumCircuit(name="bell_pair")
   sub.h(0).cx(0, 1)

   # Compose into a larger circuit
   main = QuantumCircuit(name="composed")
   main = main.compose(sub, qubits=[0, 2])   # map sub's qubits to 0,2
   main = main.compose(sub, qubits=[1, 3])   # map sub's qubits to 1,3

Compiling to DAG
-----------------

.. code-block:: python

   circ = QuantumCircuit()
   circ.h(0).cx(0, 1).rz(0.3, 0).measure_all()

   dag = circ.to_dag()

   print(f"DAG nodes: {dag.num_nodes()}")
   print(f"DAG edges: {dag.num_edges()}")
   print(f"Topological order: {dag.topological_order()}")

   # View by layers
   for i, layer in enumerate(dag.layers()):
       print(f"Layer {i}: nodes {layer}")

Next Steps
----------

- :doc:`variational` — Compile observable objectives for VQE and parameter sweeps
- :doc:`api` — Full API reference
- :doc:`architecture` — Architecture design overview
