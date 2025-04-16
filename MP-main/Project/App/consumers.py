import json
import numpy as np
import asyncio  # ✅ Fix: Import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from scipy.integrate import solve_ivp

class NBodyConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.y0 = np.array([
            0, 0, 0,  # Earth
            384400000, 0, 0,  # Moon
            200000000, 0, 0,  # Spacecraft
            0, 0, 0,  # Velocities for Earth
            0, 1022, 0,  # Velocities for Moon
            0, 1500, 0   # Velocities for Spacecraft
        ])
        self.masses = np.array([5.97e24, 7.35e22, 1000])
        self.timestep = 1000  # Time step for updates
        self.time_elapsed = 0
        await self.run_simulation()

    async def run_simulation(self):
        def equations(t, y, masses):
            num_bodies = len(masses)
            positions = y[:num_bodies * 3].reshape((num_bodies, 3))
            velocities = y[num_bodies * 3:].reshape((num_bodies, 3))
            derivatives = np.zeros_like(y)

            G = 6.67430e-11  # Gravitational constant
            accelerations = np.zeros_like(positions)

            for i in range(num_bodies):
                for j in range(num_bodies):
                    if i != j:
                        r = positions[j] - positions[i]
                        distance = np.linalg.norm(r) + 1e-10  # Avoid division by zero
                        accelerations[i] += (G * masses[j] / distance**3) * r

            derivatives[:num_bodies * 3] = velocities.flatten()
            derivatives[num_bodies * 3:] = accelerations.flatten()
            return derivatives

        while True:
            solution = solve_ivp(
                equations, 
                (self.time_elapsed, self.time_elapsed + self.timestep), 
                self.y0, 
                args=(self.masses,), 
                method="RK45", 
                t_eval=[self.time_elapsed + self.timestep]  
            )
            
            self.y0 = solution.y[:, -1]  # Update initial conditions for the next step
            self.time_elapsed += self.timestep

            positions = self.y0[:9].reshape(3, -1, 3)  # Extract 3D positions
            await self.send(json.dumps({"positions": positions.tolist()}))

            await asyncio.sleep(0.1)  # ✅ Fix: asyncio.sleep is now correctly imported
