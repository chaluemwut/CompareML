import time

class ImputerMLTest(object):

    def create_datafile_duplication(self):
        pass

    def create_datafile_missing_value(self):
        pass

    def create_datafile(self):
        self.create_datafile_duplication()
        self.create_datafile_missing_value()

if __name__ == '__main__':
    start = time.time()
    impMLTest = ImputerMLTest()
    total = (time.time()-start)/60.0
    print 'Total time {} minute'.format(total)
