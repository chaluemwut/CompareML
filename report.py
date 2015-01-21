class ReportUtil(object):
    
    @staticmethod
    def report_print(*data):
        print len(data)

if __name__ == '__main__':
    ReportUtil.report_print([1,2,3],[34,5,6])