

class MyTask:
    params = {}

    def applyDefParams(self, paramlists):
        from collections import ChainMap
        newparams = []
        for p in paramlists:
            if 'var' in p:
                newparams.append({p['var']: p['init']})
            else:
                for k in p:
                    newparams.append({k: p[k]})
                    break

        return self.applyParams(dict(ChainMap(*newparams)))

    def applyParams(self, params):
        self.params = params
        for p in params:
            self.__dict__[p] = params[p]
        return True

    def reset(self):
        pass

    def __str__(self):
        return '<' + self.__class__.__name__ + '> ' + str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def shortname(self):
        return self.__class__.__name__

    def save(self, file):
        import pickle
        file = file + '.pkl'
        with open(file, 'wb') as f:
            pickle.dump([self], f)

    def load(self, file):
        pass
