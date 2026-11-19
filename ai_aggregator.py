"""Required AI aggregator entry point. Trains models and evaluates validation-selected fusion."""
from cybersentinel.training import train
if __name__=='__main__':
    print(train())
