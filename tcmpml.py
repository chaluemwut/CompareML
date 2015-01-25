__author__ = 'Chaluemwut'
from cmpml import CmpML
import logging

logging.basicConfig(level=logging.DEBUG)

class ThreadCmpML(CmpML):

    def process_cmp_new(self):
        logging.info('start cmp..')

if __name__ == '__main__':
    obj = ThreadCmpML()
    obj.process_cmp_new()
