import sys
import pytest
if sys.version_info < (3,11): pytest.skip("rqm-compiler 0.4 requires Python 3.11+",allow_module_level=True)
import pennylane as qml
from rqm_compiler import Circuit,compile_representation_aware,lower_circuit_for_backend
from rqm_pennylane.export import compiled_circuit_to_qnode_ops

def test_compiler_0_4_candidate_exports_to_pennylane():
 c=Circuit(2);c.rxx(0,1,.2);c.ryy(0,1,.1);c.rzz(0,1,-.07)
 compiled=compile_representation_aware(c)
 ops=compiled_circuit_to_qnode_ops(compiled.circuit)
 assert len(ops)==3
 dev=qml.device("default.qubit",wires=2)
 @qml.qnode(dev)
 def run():
  for op in ops:op()
  return qml.expval(qml.PauliZ(0))
 assert isinstance(float(run()),float)

def test_compiler_report_is_out_of_band_from_pennylane_descriptor_bridge():
 c=Circuit(1);c.rx(0,.2)
 compiled=compile_representation_aware(c)
 assert compiled.report.representation_complexity is not None
 lowered=lower_circuit_for_backend(compiled.circuit,backend_family="braket_gate_model")
 ops=compiled_circuit_to_qnode_ops(lowered)
 assert len(ops)>=1
