from matplotlib.font_manager import X11FontDirectories
import numpy as np
import matplotlib.pyplot as plt

opcje = ['p','k','n']

szanse = np.array([[1/3, 1/3, 1/3],
                  [1/3, 1/3, 1/3],
                  [1/3, 1/3, 1/3]])

alfa = 0.1

punkty = [0]

def bot(ile):
  botszanse = np.array([[0.2, 0.7, 0.1],
                        [0.4, 0.3, 0.3],
                        [0.2, 0.2, 0.6]])
  punkty = [0]
  i=0
  x = ''
  z = '' 
  b = 'p'
  while i<ile:

    if (x == ''):
      a = np.random.choice(['p','k','n'] , p=[1/3,1/3,1/3])
      b = np.random.choice(['p','k','n'] , p=botszanse[0])

      if((a == 'p' and b == 'n') or (a == 'k' and b == 'p') or (a == 'n' and b == 'k')):
        punkty[i]=1
      if((a == 'p' and b == 'k') or (a == 'k' and b == 'n') or (a == 'n' and b == 'p')):
        punkty[i]=-1

      x = b
      z = a
      i += 1

    else:
      a = np.random.choice(['p','k','n'] , p=szanse[getIndex(x)])   
      b = np.random.choice(['p','k','n'] , p=botszanse[getIndex(z)])

      uczenie(x,b)

      if(a == b):
        punkty.append(punkty[i-1])        
      if((a == 'p' and b == 'n') or (a == 'k' and b == 'p') or (a == 'n' and b == 'k')):
        punkty.append(punkty[i-1]+1)
      if((a == 'p' and b == 'k') or (a == 'k' and b == 'n') or (a == 'n' and b == 'p')):
        punkty.append(punkty[i-1]-1)

      x = b
      z = a
      i += 1
  print(szanse)
  print(punkty)
  plt.plot(punkty)
  plt.ylabel('Punkty')
  plt.xlabel('Gra')
  plt.show()


def getIndex(x):
    if(x == 'p'):
        return 0
    if(x == 'k'):
        return 1
    if(x == 'n'):
        return 2

def uczenie(x, y):

  i=getIndex(x)

  if(y == 'p') and (szanse[i][2]<1-alfa) and (szanse[i][0]>0+alfa/2) and (szanse[i][1]>0+alfa/2):
      szanse[i][0] -= alfa/2
      szanse[i][1] -= alfa/2
      szanse[i][2] += alfa

  if(y == 'k')  and (szanse[i][0]<1-alfa) and (szanse[i][1]>0+alfa/2) and (szanse[i][2]>0+alfa/2):
      szanse[i][0] += alfa
      szanse[i][1] -= alfa/2
      szanse[i][2] -= alfa/2
  if(y == 'n') and (szanse[i][1]<1-alfa) and (szanse[i][0]>0+alfa/2) and (szanse[i][2]>0+alfa/2):
      szanse[i][0] -= alfa/2
      szanse[i][1] += alfa
      szanse[i][2] -= alfa/2

 

x = ''
b = 'p'

#bot(500)6

while (b == 'p') or (b == 'k') or (b == 'n'):

    i=0

    if (x == ''):
      a = np.random.choice(['p','k','n'] , p=[1/3,1/3,1/3])
      b = input('trzy czte... ry (p,k,n)')

      if(a == b):
        print('remis')
      if((a == 'p' and b == 'n') or (a == 'k' and b == 'p') or (a == 'n' and b == 'k')):
        print('wygrana')
        punkty[i]=1
      if((a == 'p' and b == 'k') or (a == 'k' and b == 'n') or (a == 'n' and b == 'p')):
        print('przegrana')
        punkty[i]=-1

      x = b
      i += 1

    else:
      #index = np.argmax(szanse[getIndex(x)])
      #if(index == 0):
      #   a = 'p'
      #if(index == 1):
      #  a = 'k'
      #if(index == 2):
      #  a = 'n'

      a = np.random.choice(['p','k','n'] , p=szanse[getIndex(x)])
      
      b = input('trzy czte... ry (p,k,n)')

      uczenie(x,b)

      if(a == b):
        print('remis')
        punkty.append(punkty[i-1])
        
      if((a == 'p' and b == 'n') or (a == 'k' and b == 'p') or (a == 'n' and b == 'k')):
        print('wygrana')
        punkty.append(punkty[i-1]+1)
      if((a == 'p' and b == 'k') or (a == 'k' and b == 'n') or (a == 'n' and b == 'p')):
        print('przegrana')
        punkty.append(punkty[i-1]-1)

      x = b
      i += 1
print(szanse)
print(punkty)
plt.plot(punkty)
plt.ylabel('Punkty')
plt.xlabel('Gra')
plt.show()