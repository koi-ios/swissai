import pytest

from pystartup import cudafacts as cf

DEVS = [
    {"index": 0, "major": 9, "minor": 0, "name": "H100", "uuid": "GPU-aaaa-0000"},
    {"index": 1, "major": 9, "minor": 0, "name": "H100", "uuid": "GPU-bbbb-1111"},
    {"index": 2, "major": 9, "minor": 0, "name": "H100", "uuid": "GPU-cccc-2222"},
]


def test_visible_devices_semantics():
    assert cf.visible_devices(DEVS, None) == DEVS
    assert cf.visible_devices(DEVS, "") == []
    assert [d["index"] for d in cf.visible_devices(DEVS, "2,0")] == [2, 0]
    assert [d["index"] for d in cf.visible_devices(DEVS, "0,7,1")] == [0]  # stops at the first invalid
    assert [d["index"] for d in cf.visible_devices(DEVS, "GPU-bbbb")] == [1]
    assert cf.visible_devices(DEVS, "MIG-1234") is None


def test_heterogeneous_node_gives_no_facts():
    devs = DEVS[:1] + [dict(DEVS[1], major=8)]
    assert cf.facts_from_devices(devs, "t") is None
    f = cf.facts_from_devices(DEVS[:2], "t")
    assert (f.count, f.major, f.minor, f.name) == (2, 9, 0, "H100")


def test_cudatouch_is_not_an_exception():
    assert not issubclass(cf.CudaTouch, Exception)
    with pytest.raises(cf.CudaTouch):
        try:
            raise cf.CudaTouch("x")
        except Exception:  # library code like this must not swallow it
            pass


@pytest.fixture
def armed_guard():
    pytest.importorskip("torch")
    g = cf.CudaGuard(cf.CudaFacts(count=2, major=9, minor=0, name="NVIDIA H100 80GB HBM3")).arm()
    yield g
    g.disarm()


def test_guard_serves_facts(armed_guard):
    import torch

    assert torch.cuda.is_available() is True
    assert torch.cuda.device_count() == 2
    assert torch.cuda.current_device() == 0
    assert torch.cuda.get_device_capability() == (9, 0)
    assert torch.cuda.get_device_capability(1) == (9, 0)
    p = torch.cuda.get_device_properties(0)
    assert (p.major, p.minor, p.name) == (9, 0, "NVIDIA H100 80GB HBM3")


def test_guard_trips_on_everything_else(armed_guard):
    import torch

    with pytest.raises(cf.CudaTouch):
        torch.cuda.get_device_properties(0).multi_processor_count
    with pytest.raises(cf.CudaTouch):
        torch.cuda._lazy_init()
    with pytest.raises(cf.CudaTouch):
        torch.cuda.mem_get_info()
    with pytest.raises(cf.CudaTouch):
        torch.cuda.get_device_capability(5)  # beyond the visible devices
    expected = 4
    if hasattr(torch._C, "_cuda_getDeviceCount"):  # CUDA builds only
        with pytest.raises(cf.CudaTouch):
            torch._C._cuda_getDeviceCount()
        expected += 1
    assert len(armed_guard.trips) == expected


def test_disarm_restores_and_captured_references_delegate():
    torch = pytest.importorskip("torch")
    real_available = torch.cuda.is_available()
    g = cf.CudaGuard(cf.CudaFacts(count=2, major=9, minor=0, name="X")).arm()
    captured = torch.cuda.is_available          # a module did `from torch.cuda import is_available`
    props = torch.cuda.get_device_properties(0)  # and kept a properties object
    g.disarm()
    assert torch.cuda.is_available() == real_available
    assert captured() == real_available         # fake now delegates to the real function
    if not real_available:
        with pytest.raises(Exception) as ei:     # real lookup, not a CudaTouch
            props.multi_processor_count
        assert not isinstance(ei.value, cf.CudaTouch)


def test_no_facts_means_every_query_trips():
    torch = pytest.importorskip("torch")
    g = cf.CudaGuard(None).arm()
    try:
        with pytest.raises(cf.CudaTouch):
            torch.cuda.is_available()
        with pytest.raises(cf.CudaTouch):
            torch.cuda.device_count()
    finally:
        g.disarm()


def test_zero_device_facts():
    torch = pytest.importorskip("torch")
    g = cf.CudaGuard(cf.CudaFacts(count=0, major=0, minor=0, name="none")).arm()
    try:
        assert torch.cuda.is_available() is False
        with pytest.raises(cf.CudaTouch):
            torch.cuda.current_device()
        with pytest.raises(cf.CudaTouch):
            torch.cuda.get_device_capability()
    finally:
        g.disarm()


def test_invariant_helpers_run_here():
    assert cf.nvidia_fds() == []  # this test machine has no NVIDIA driver handles open
    assert cf.torch_cuda_initialized() in (False, None)
