import sys, time, importlib, traceback
sys.path.insert(0, "/Users/koiios/Documents/Projects/swissai/opportunities/08-preshard-roundtrip/vendor/sglang/python")
t=time.time()
for mod in sys.argv[1:]:
    try:
        importlib.import_module(mod)
        print("OK", mod, round(time.time()-t,1))
    except Exception as e:
        print("FAIL", mod, type(e).__name__, str(e)[:300])
        traceback.print_exc(limit=3)
