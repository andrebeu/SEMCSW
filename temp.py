import os
import sys
import traceback
from functools import wraps
from multiprocessing import Process, Queue

class Processify(object):
    def __init__(self, func):
        self.func = func
        
    def process_func(q, *args, **kwargs):
        try:
            ret = self.func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))
        
    def run(self, *args, **kwargs):
        q = Queue()
        p = Process(target=self.process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()
        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret

def processify(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        C = Processify(func)
        ret = C.run(*args, **kwargs)
        return ret
    return wrapper


# def processify(func):
#     '''Decorator to run a function as a process.
#     Be sure that every argument and the return value
#     is *pickable*.
#     The created process is joined, so the code does not
#     run in parallel.

#     Credit: I took this function from Marc Schlaich's github:
#     https://gist.github.com/schlamar/2311116
#     '''

#     def process_func(q, *args, **kwargs):
#         try:
#             ret = func(*args, **kwargs)
#         except Exception:
#             ex_type, ex_value, tb = sys.exc_info()
#             error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
#             ret = None
#         else:
#             error = None

#         q.put((ret, error))

#     # register original function with different name
#     # in sys.modules so it is pickable
#     process_func.__name__ = func.__name__ + 'processify_func'
#     setattr(sys.modules[__name__], process_func.__name__, process_func)

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         q = Queue()
#         p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
#         p.start()
#         ret, error = q.get()
# #         p.join()

#         if error:
#             ex_type, ex_value, tb_str = error
#             message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
#             raise ex_type(message)

#         return ret
#     return wrapper

           
@processify
def test_function():
    return os.getpid()
           

if __name__ == "__main__":
#     print(main())
    print(test_function())
           
      