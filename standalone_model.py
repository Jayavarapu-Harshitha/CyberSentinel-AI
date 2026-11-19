"""Required standalone AI model entry point."""
from cybersentinel.training import train
if __name__=='__main__':
    print(train(models=['mlp']))
