import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class WaveFunction:

    def __init__(self, psi, m, hbar, V=None, x0=-10, xf=10, n_points=2000):
        self.m = m
        self.hbar = hbar
        self.V = lambda x: V(x) if V != None else 0

        self.domain = np.linspace(x0, xf, n_points)
        self.dx = self.domain[1] - self.domain[0]
        self.funcImage = psi(self.domain)

    def __call__(self):
        return self.funcImage.copy()
        
    def discreteNormalize(self):
        self.funcImage /= np.sqrt(sum(abs(self.funcImage)**2) * self.dx)
    
    def discreteLaplacian(self, funcImage):
        diff = np.zeros_like(funcImage)
        diff[1:-1] = funcImage[2:] - 2*funcImage[1:-1] + funcImage[:-2]
        return diff
    
    def schrodinger(self, funcImage):
        return 1j*self.hbar/2*self.m * self.discreteLaplacian(funcImage) - 1j*self.V(self.domain)*funcImage
    

class WaveFunctionIntegrator:

    def __init__(self, initWaveFunc):
        self.waveFunc = initWaveFunc
        self.waveFuncStates = [initWaveFunc()]

    def euler_step(self, dt):
        self.waveFunc.funcImage += self.waveFunc.schrodinger(self.waveFunc()) * dt
    
    def RK4_step(self, dt):
        k1 = dt * self.waveFunc.schrodinger(self.waveFunc())
        k2 = dt * self.waveFunc.schrodinger(self.waveFunc() + k1/2)
        k3 = dt * self.waveFunc.schrodinger(self.waveFunc() + k2/2)
        k4 = dt * self.waveFunc.schrodinger(self.waveFunc() + k3)
        self.waveFunc.funcImage += 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def integrate(self, steps, dt, append_every=10, method="RK4"):
        if method == "RK4":
            for t in range(steps):
                self.RK4_step(dt)
                self.waveFunc.discreteNormalize()

                if t%append_every == 0:
                    self.waveFuncStates.append(self.waveFunc())
        
        if method == "euler":
            for t in range(steps):
                self.euler_step(dt)
                self.waveFunc.discreteNormalize()

                if t%append_every == 0:
                    self.waveFuncStates.append(self.waveFunc())
        
        return self.waveFuncStates
    
    def animate(self, **kwargs):

        X = [(abs(self.waveFuncStates[i]))*np.exp(np.arctan2(self.waveFuncStates[i].imag, self.waveFuncStates[i].real)*1j) for i in range(len(self.waveFuncStates))]

        plt.rcParams['axes.titlepad'] = -14
        plt.rcParams["toolbar"] = "None"
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes((0.05,0.05,0.9,0.9), projection="3d", aspect="equal")
     
        ax.set_title("Particle in 1D")
        ax.set_xlim(kwargs["ylim"])
        ax.set_ylim(kwargs["xlim"])
        ax.set_zlim(kwargs["ylim"])

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False

        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_zticks([])
        
        real, = ax.plot3D([],[],[],c="magenta", alpha=0.75)
        psi_abs, = ax.plot3D([],[],[],c="white")
        
        ZEROS = np.zeros_like(self.waveFunc.domain)
        ax.plot3D(ZEROS, self.waveFunc.domain, ZEROS, c="white",alpha=0.25)

        def _animation_func(frame):

            ax.view_init(10, frame/4 - 45)

            real.set_data_3d(X[frame].imag, self.waveFunc.domain, self.waveFuncStates[frame].real)
            psi_abs.set_data_3d(ZEROS, self.waveFunc.domain, abs(self.waveFuncStates[frame]))

            return psi_abs,real,

        anim = FuncAnimation(fig, _animation_func, len(self.waveFuncStates), interval=1, blit=False)
        plt.show()

    

m = 20
p = 30
hbar = 1
mu = -0.85
sigma = 0.5

k_spring = 3

potential_barrier1 = lambda x: np.where((x>0.5)&(x<0.7), 0.9, 0)
potential_barrier2 = lambda x: np.where((x<-0.5)&(x>-0.7), 1.3, 0)
potential_barrier = lambda x: potential_barrier1(x) + potential_barrier2(x)
V = lambda x: 0.5*k_spring*x*x

psi = WaveFunction(lambda x: np.exp(-(x-mu)**2 / (2*sigma**2), dtype=complex) * np.exp(1j*p*x),
                   m, hbar, V=potential_barrier1, n_points=2000)


solver = WaveFunctionIntegrator(psi)
states = solver.integrate(steps=7000,dt=0.01, append_every=15, method="RK4")
# states = solver.integrate(steps=7000,dt=-0.01j, append_every=15, method="RK4") # <- Lowest energy eigenstate
solver.animate(xlim=(-1.75,1.75), ylim=(-1.75,1.75))
