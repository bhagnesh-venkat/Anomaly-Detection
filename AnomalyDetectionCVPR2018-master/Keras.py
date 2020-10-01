from sklearn.preprocessing import MinMaxScaler
from keras.layers          import Dense, Dropout
from keras.models          import Sequential
from keras.optimizers      import RMSprop
from matplotlib            import pyplot as plt
from numpy.random          import random
#from keras.utils import plot_model
#from IPython.display import SVG
from keras.utils import plot_model
from ann_visualizer.visualize import ann_viz

def baseline_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=1))
    #model.add(Dense(64, activation='relu', input_dim=1))
    model.add(Dense( 1, activation='linear'))
    optimizer = RMSprop(0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def f_lin(x):
    a,b = 1,0.5
    return a*x+b

def f_quad(x):
    a,b,c = 1.,0.5,1.
    return a*pow(x,2.)+b*x+c

def f_circ(x):
    from numpy import sqrt
    r2 = 4
    return sqrt(4-x*x)

def f_sin1(x):
    from numpy import sin, pi
    return sin(pi*x/2.)

def f_sin2(x):
    from numpy import sin
    return sin(10*x)

def natalia(fufu,lab):
    # Regresión de Natalia
    # El argumento fufu es la funcion
    # para generar los 'y's de entrenamiento
    # e.g.  linear, cuadratica, etc

    # Generate dummy data
    ntrain, ntest = 200, 20
    # training
    x_train = random((ntrain, 1))
    scaler  = MinMaxScaler(feature_range=(-1, 1))
    scaler  = scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    y_train = fufu(x_train)+0.1*random((ntrain, 1)) #y_1 = a*x_train+b

    #Testig
    x_test  = .5+random((ntest, 1))
    #scaler  = MinMaxScaler(feature_range=(-1, 1)) # commented to allow extrapolation
    #scaler  = scaler.fit(x_test)
    x_test  = scaler.transform(x_test)
    model1  = baseline_model()
    plot_model(model1, to_file='model.png', show_shapes = True)
    ann_viz(model1, title='model')
    #model1.fit(x_train, y_train, epochs=40, verbose=0, validation_split=0.3)
    #SVG(model_to_dot(model1).create(prog='dot', format='svg'))

    y       = model1.predict(x_train)
    y_test  = model1.predict(x_test)

    # plotting
    xmin    = min([x_train.min(),x_test.min()])
    xmax    = max([x_train.max(),x_test.max()])
    ymin    = min([y_train.min(),y_test.min(), y.min()])
    ymax    = max([y_train.max(),y_test.max(), y.max()])

    f   = plt.figure()
    ax1 = f.add_subplot(311)
    ax1.set_title(lab)
    ax1.plot(x_train,y_train,'.', label = 'train')
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])
    ax1.legend()
    ax1.grid()

    ax2 = f.add_subplot(312)
    ax2.plot(x_train,y_train,'.', label='train')
    ax2.plot(x_train, y , '.', label='pred on training set')
    ax2.legend()
    ax2.set_xlim([xmin,xmax])
    ax2.set_ylim([ymin,ymax])
    ax2.legend()
    ax2.grid()

    ax3 = f.add_subplot(313)
    ax3.plot(x_train,y_train,'.', label='train')
    ax3.plot(x_test, y_test , '*', label='test')
    ax3.set_xlim([xmin,xmax])
    ax3.set_xlim([xmin,xmax])
    ax3.set_ylim([ymin,ymax])
    ax3.legend()
    ax3.grid()

    plt.show()
    plt.close()

# Main
funcs = [f_lin,     f_quad,       f_circ,         f_sin1,      f_sin2]
labs  = ['linear', 'quadratic', 'circular arc',  'sine_short',  'sine_long']
for fufu,lab in zip(funcs,labs): natalia(fufu,lab)