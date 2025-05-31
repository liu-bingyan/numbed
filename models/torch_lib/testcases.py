import torch
import torch.nn as nn
from models._numerical_embedding import NumericalEmbedding
#import configargparse

#parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser,
#                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--case', type=str, required=True)
#TESTCASE = parser.parse_args()
TESTCASE = 'BIN'
if TESTCASE == "BIN":
    model = NumericalEmbedding(3, 3, use_M_matrix=False,num_idx=[2],initialization='alpha')
    x = torch.tensor([[1, 2, 1/3],
                       [4, 5, 1/6],
                       [7,8,1/9],
                       [10,11,1/12]], dtype=torch.float32)
    y = model(x)
    print(y)
    print(model.state_dict())
   
if TESTCASE == "PAREN":
    act1 = nn.Tanh()
    print(type(act1))
    act2 = nn.Tanh
    print(type(act2))

if TESTCASE == 0:
    tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    sliced_tensor = tensor[1, :]  
    print(sliced_tensor.shape)  # Output: torch.Size([2, 3])

if TESTCASE == 1:

    num_features = 2
    num_bins = 3
    x1 = torch.zeros(5, 1)
    x2 = torch.tensor([[3**i] for i in range(5)])
    x = torch.cat((x1, x2), dim=1)
    model = NumericalEmbedding(num_features, num_bins, use_M_matrix=True)
    y = model(x)
    print(y)
    param = dict(model.named_parameters())
    for k, v in param.items():
        print(k, v)

if TESTCASE == 2:
    n = 10
    m1 = MLP(input_size=1, hidden_size=10, output_size=1)
    m2 = MLP(input_size=1, hidden_size=10, output_size=1)

    X = torch.tensor([i for i in range(n)])
    y = torch.tensor([2*i for i in range(n)])


    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(m1.parameters(), lr=0.01)

    for epoch in range(100):
        # Forward pass
        outputs = m2(m1(X))
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}')
    
if TESTCASE == 3:
    def foo(bar, baz=1):
        if baz == 1:
            bar = []
        if baz == 2:
            bar[0] = 0
        return bar    
    bar = [1]
    newbar = foo(bar,2)
    print(bar,newbar)

if TESTCASE == 4:
          
    from collections import deque
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
        def __str__(self):
            if self.next == None:
                return str(self.val)
            return str(self.val)+","+str(self.next)
 
    class Solution:
        def removeNthFromEnd(self, head, n: int):
            q = deque()
            counter = 0
            curr = head
            while counter<n+1:
                q.append(curr)
                curr = curr.next
                counter += 1
                if curr == None:
                    if counter == n:
                        return head.next
                    else:
                        return head

            while curr != None:
                q.append(curr)
                q.popleft()
                curr = curr.next

            prec = q.popleft()
            if n == 1:
                prec.next = None
            else: # queue has at least 3 elements
                q.popleft()
                succ = q.popleft()
                prec.next = succ
            return head
        
    head = ListNode(1,ListNode(2,ListNode(3,ListNode(4,ListNode(5,None)))))
    sol = Solution()
    ans = sol.removeNthFromEnd(head,1)
    print(ans)
  
if TESTCASE == 'INITSUB':
    '''
    call super().__init_subclass__()
            inside __init_subclass__
            one need to modify the init of the subclass
            and define what want to do next # the purpose
    the (new) init of the subclass will be called automatically
    '''
    class A:
        def __init_subclass__(cls):
            def new_init(self, args,init=cls.__init__,**kwdarg):
                init(self, args,**kwdarg)
                self.postinit()
                print(f' print foo : {self.foo()}')
            #print('Call __init_subclass__')
            super().__init_subclass__()
            #print('Modifying init of B')
            cls.__init__ = new_init
            #print('Calling the new init')
        def __init__(self,args):
            self.args = args
        def postinit(self):
            print('Post Init of A')
        def foo(self):
            print('bar')
       
    class B(A):
        def __init__(self,args,key) -> None:
            #print('Init of B')
            super().__init__(args)
            self.key = key
            print(self.args)
            print(self.key)
        def foo(self):
            print('barplus')

    b = B(1,key=2)
    b.foo()

if TESTCASE == 'INHERIT':
    class Parent:
        def __init__(self):
            self.public_var = "I'm public"
            self._protected_var = "I'm protected"
            self.__private_var = "I'm private"

    class Child(Parent):
        def __init__(self):
            super().__init__()
            print(self.public_var)  # Accessing public member 
            print(self._protected_var)  # Accessing protected member

            # This will likely raise an AttributeError 
            # print(self.__private_var)  
    a = Child()

if TESTCASE == 'STATEDICT':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    net = Net()
    print(net)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Print model's state_dict
    print(f"Model's state_dict: {type(net.state_dict())}")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    print()

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

if TESTCASE == 'STROVERFLOW':
    s = "123"
    print(s[3])

if TESTCASE == 'CLONE':
    class A():
        cloned = False
        def __init__(self):
            pass
        def clone(self):
            A.cloned = True
            return self.__class__()
        def is_cloned(self):
            return A.cloned
    class B(A):
        def __init__(self):
            super().__init__()
        def clone(self):
            A.cloned = True
            return self.__class__()

    b = B()
    print(b.is_cloned())
    b.clone()
    print(b.is_cloned())
    b2 = B()
    print(b.is_cloned())
