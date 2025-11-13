import random

class PSO:
    def __init__(self, func, bounds, num_particles=30, iterations=100,
                 w=0.7, c1=1.5, c2=1.5):
        self.func = func
        self.bounds = bounds
        self.num_particles = num_particles
        self.iterations = iterations
        self.w, self.c1, self.c2 = w, c1, c2

    def optimize(self):
        dim = len(self.bounds)
        particles = []

        for _ in range(self.num_particles):
            pos = [random.uniform(a, b) for a, b in self.bounds]
            vel = [random.uniform(-1, 1) for _ in range(dim)]
            val = self.func(*pos)
            particles.append({
                "pos": pos,
                "vel": vel,
                "pbest": pos[:],
                "pbest_val": val
            })

        gbest = min(particles, key=lambda p: p["pbest_val"])
        gbest_pos, gbest_val = gbest["pbest"][:], gbest["pbest_val"]

        for t in range(self.iterations):
            for p in particles:
                r1, r2 = random.random(), random.random()

                for i in range(dim):
                    p["vel"][i] = (
                        self.w * p["vel"][i]
                        + self.c1 * r1 * (p["pbest"][i] - p["pos"][i])
                        + self.c2 * r2 * (gbest_pos[i] - p["pos"][i])
                    )
                    p["pos"][i] += p["vel"][i]
                    p["pos"][i] = max(self.bounds[i][0], min(self.bounds[i][1], p["pos"][i]))

                val = self.func(*p["pos"])
                if val < p["pbest_val"]:
                    p["pbest_val"] = val
                    p["pbest"] = p["pos"][:]

            best = min(particles, key=lambda p: p["pbest_val"])
            if best["pbest_val"] < gbest_val:
                gbest_pos, gbest_val = best["pbest"][:], best["pbest_val"]

        return gbest_pos, gbest_val