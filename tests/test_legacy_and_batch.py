import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_legacy_single_qubit_and_flat_column_inputs():
    m = load(ROOT/'QAOA_class.py', 'legacy_qaoa')
    h = np.array([.3,-.7])
    angles = [.2,-.4,.7,.1]
    q = m.QAOA(2,h)
    x = np.array([[0,1],[1,0]])
    u = np.eye(2, dtype=complex)
    for g,b in zip(angles[:2],angles[2:]):
        u = expm(-1j*b*x)@np.diag(np.exp(-1j*g*h))@u
    plus = np.ones(2)/np.sqrt(2)
    np.testing.assert_allclose(q.qaoa_ansatz(angles).ravel(), u@plus, atol=1e-14)
    np.testing.assert_allclose(q.apply_ansatz(angles,plus), u@plus, atol=1e-14)
    np.testing.assert_allclose(q.apply_ansatz(angles,plus[:,None]), (u@plus)[:,None], atol=1e-14)


def test_batch_entrypoint_import_seed_and_shared_kernel(tmp_path):
    m = load(ROOT/'Zhores/QAOA_locality_Zhores.py', 'zhores_qaoa')
    a,_,beta,gamma=m.compiler(3,2,2,2,seed=13)
    b,_,beta2,gamma2=m.compiler(3,2,2,2,seed=13)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(beta,beta2)
    np.testing.assert_array_equal(gamma,gamma2)
    assert m.count_solutions_XI(2,np.zeros((4,4))) == ([],0,0.,0)
    output = m.main(['2','2','2','2',str(tmp_path),'213'])
    assert output.name == '2n_2p_2t_2l_13seed.csv'
    lines=output.read_text().splitlines()
    assert len(lines) == 6 and lines[0] == '2,2,2,2'


def test_legacy_layerwise_optimized_start_and_evaluation_count(monkeypatch):
    m=load(ROOT/'QAOA_class.py','legacy_layerwise')
    original=m.minimize
    evaluations=[]
    def counted(*args,**kwargs):
        result=original(*args,**kwargs)
        evaluations.append(result.nfev)
        return result
    monkeypatch.setattr(m,'minimize',counted)
    q=m.QAOA(2,np.array([1.,-1.]))
    q.heruistic_LW_seed1=3; q.heruistic_LW_seed2=2
    q.run_heuristic_LW(seed=12)
    assert q.opt_iter==sum(evaluations)+1
    assert q.q_energy < -.99999
    assert abs(q.expectation(q.opt_angles)-q.q_energy)<1e-14
    p1=m.QAOA(1,np.array([1.,-1.])); p1.heruistic_LW_seed1=3
    p1.run_heuristic_LW(seed=12)
    assert p1.q_energy < -.99999
