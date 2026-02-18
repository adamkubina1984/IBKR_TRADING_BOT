from ib_insync import IB
import sys

ib = IB()
try:
    ib.connect('127.0.0.1', 7496, clientId=1)
    print('CONNECTED' if ib.isConnected() else 'NOT_CONNECTED')
    ib.disconnect()
except Exception as e:
    print('ERROR', e)
    sys.exit(2)
