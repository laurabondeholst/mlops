import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

from torch import nn, optim
import torch

from data import mnist, mnist_loader
from model import MyAwesomeModel

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        # parser = argparse.ArgumentParser(
        #     description="Script for either training or evaluating",
        #     usage="python main.py <command>"
        # )
        # parser.add_argument("-t", "--train", help="Train network")
        # parser.add_argument("-e", "--eval", help="Evaluate network")
        # print(sys.argv[1:2])
        # print(parser.print_help())
        # args = parser.parse_args(sys.argv[1:2])
        # # print(args.command)

    
        # # if not hasattr(self, args.eval, args.train):
        # #     print('Unrecognized command')
            
        # #     parser.print_help()
        # #     exit(1)
        # # use dispatch pattern to invoke method with same name
        # getattr(self, args.train)()
        # # getattr(self, args.eval)()
        self.evaluate()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        # parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        #  Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_set, _ = mnist()
        train_set, _ = mnist_loader(train_set, _)
        n = len(train_set)

        
        epochs = 30
        steps = 0

        train_loss = []
        for e in range(epochs):
            print(f'Running epoch {e}')
            running_loss = 0
            accuracy = []
            for images, labels in train_set:
                
                optimizer.zero_grad()
                
                log_ps = model(images)

                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                
            else:
                train_loss.append(running_loss)
        
        torch.save(model.state_dict(), 'checkpoint.pth')

        x = range(0,epochs)
        plt.plot(x, train_loss)



    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        # parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        # args = parser.parse_args(sys.argv[2:])
        # print(args)
        
        # TODO: Implement evaluation logic here
        # if args.load_model_from:
        #     model = torch.load(args.load_model_from)

        model = MyAwesomeModel()
        state_dict = torch.load('checkpoint.pth')
        model.load_state_dict(state_dict)

        _, test_set = mnist()
        _, test_set = mnist_loader(_, test_set)
        n = len(test_set)*64

        accuracy = []
        for images, labels in test_set:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy.append(torch.sum(equals))
            
        else:
            accuracy  = np.array(accuracy)
            print(accuracy)
            print(n)
            a = np.sum(accuracy)/n
            print(f'Accuracy: {a*100}%')
                

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    