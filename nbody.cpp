#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct simulation {
    size_t nbpart;

    // Host arrays
    std::vector<double> mass;
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> fx, fy, fz;

    // Device arrays
    double *d_mass, *d_x, *d_y, *d_z;
    double *d_vx, *d_vy, *d_vz;
    double *d_fx, *d_fy, *d_fz;

    simulation(size_t nb)
      : nbpart(nb), mass(nb),
        x(nb), y(nb), z(nb),
        vx(nb), vy(nb), vz(nb),
        fx(nb), fy(nb), fz(nb),
        d_mass(nullptr), d_x(nullptr), d_y(nullptr), d_z(nullptr),
        d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
        d_fx(nullptr), d_fy(nullptr), d_fz(nullptr) {}

    void allocate_device_memory() {
        size_t nbytes = nbpart * sizeof(double);
        cudaMalloc(&d_mass, nbytes);
        cudaMalloc(&d_x, nbytes); cudaMalloc(&d_y, nbytes); cudaMalloc(&d_z, nbytes);
        cudaMalloc(&d_vx, nbytes); cudaMalloc(&d_vy, nbytes); cudaMalloc(&d_vz, nbytes);
        cudaMalloc(&d_fx, nbytes); cudaMalloc(&d_fy, nbytes); cudaMalloc(&d_fz, nbytes);
    }

    void free_device_memory() {
        cudaFree(d_mass);
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
    }

    void copy_to_device() {
        size_t nbytes = nbpart * sizeof(double);
        cudaMemcpy(d_mass, mass.data(), nbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), nbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y.data(), nbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, z.data(), nbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vx, vx.data(), nbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vy, vy.data(), nbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vz, vz.data(), nbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fx, fx.data(), nbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fy, fy.data(), nbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fz, fz.data(), nbytes, cudaMemcpyHostToDevice);
    }

    void copy_to_host() {
        size_t nbytes = nbpart * sizeof(double);
        cudaMemcpy(mass.data(), d_mass, nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(x.data(), d_x, nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(y.data(), d_y, nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(z.data(), d_z, nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(vx.data(), d_vx, nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(vy.data(), d_vy, nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(vz.data(), d_vz, nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(fx.data(), d_fx, nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(fy.data(), d_fy, nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(fz.data(), d_fz, nbytes, cudaMemcpyDeviceToHost);
    }
};

__global__ void compute_forces(int nbpart, double *mass, double *x, double *y, double *z,
                                double *fx, double *fy, double *fz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbpart) return;

    double G = 6.674e-11;
    double softening = 0.1;

    double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;

    for (int j = 0; j < nbpart; j++) {
        if (i == j) continue;

        double dx = x[j] - x[i];
        double dy = y[j] - y[i];
        double dz = z[j] - z[i];
        double distSqr = dx*dx + dy*dy + dz*dz + softening;
        double invDist = rsqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;

        double F = G * mass[i] * mass[j] * invDist3;

        fx_i += F * dx;
        fy_i += F * dy;
        fz_i += F * dz;
    }

    fx[i] = fx_i;
    fy[i] = fy_i;
    fz[i] = fz_i;
}

__global__ void update_particles(int nbpart, double *mass,
                                 double *x, double *y, double *z,
                                 double *vx, double *vy, double *vz,
                                 double *fx, double *fy, double *fz,
                                 double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbpart) return;

    vx[i] += fx[i] / mass[i] * dt;
    vy[i] += fy[i] / mass[i] * dt;
    vz[i] += fz[i] / mass[i] * dt;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

void dump_state(const simulation& s) {
    for (size_t i = 0; i < s.nbpart; i++) {
        std::cout << s.mass[i] << "\t"
                  << s.x[i] << "\t" << s.y[i] << "\t" << s.z[i] << "\t"
                  << s.vx[i] << "\t" << s.vy[i] << "\t" << s.vz[i] << "\n";
    }
}

void random_init(simulation& s) {
    for (size_t i = 0; i < s.nbpart; i++) {
        s.mass[i] = 1.0 + static_cast<double>(rand()) / RAND_MAX;
        s.x[i] = static_cast<double>(rand()) / RAND_MAX * 100.0;
        s.y[i] = static_cast<double>(rand()) / RAND_MAX * 100.0;
        s.z[i] = static_cast<double>(rand()) / RAND_MAX * 100.0;
        s.vx[i] = s.vy[i] = s.vz[i] = 0.0;
        s.fx[i] = s.fy[i] = s.fz[i] = 0.0;
    }
}

void init_solar(simulation& s) {
    if (s.nbpart < 2) return;
    s.mass[0] = 1.989e30; // Sun mass
    s.x[0] = 0.0; s.y[0] = 0.0; s.z[0] = 0.0;
    s.vx[0] = s.vy[0] = s.vz[0] = 0.0;

    s.mass[1] = 5.972e24; // Earth mass
    s.x[1] = 1.496e11; s.y[1] = 0.0; s.z[1] = 0.0;
    s.vx[1] = 0.0; s.vy[1] = 29780.0; s.vz[1] = 0.0;
}

void load_from_file(simulation& s, const std::string& filename) {
    std::ifstream file(filename);
    for (size_t i = 0; i < s.nbpart; ++i) {
        file >> s.mass[i] >> s.x[i] >> s.y[i] >> s.z[i]
             >> s.vx[i] >> s.vy[i] >> s.vz[i];
        s.fx[i] = s.fy[i] = s.fz[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " nbstep dt printevery init_type nbpart [filename]\n";
        return 1;
    }

    size_t nbstep = std::stoul(argv[1]);
    double dt = std::stod(argv[2]);
    size_t printevery = std::stoul(argv[3]);
    std::string init_type = argv[4];
    size_t nbpart = std::stoul(argv[5]);

    simulation s(nbpart);

    if (init_type == "random") {
        random_init(s);
    } else if (init_type == "solar") {
        init_solar(s);
    } else if (init_type == "file") {
        if (argc < 7) {
            std::cerr << "Missing filename for 'file' init.\n";
            return 1;
        }
        load_from_file(s, argv[6]);
    } else {
        std::cerr << "Unknown init_type: " << init_type << "\n";
        return 1;
    }

    s.allocate_device_memory();
    s.copy_to_device();

    int threads_per_block = 256;
    int blocks = (s.nbpart + threads_per_block - 1) / threads_per_block;

    for (size_t step = 0; step < nbstep; ++step) {
        compute_forces<<<blocks, threads_per_block>>>(s.nbpart, s.d_mass, s.d_x, s.d_y, s.d_z,
                                                      s.d_fx, s.d_fy, s.d_fz);
        cudaDeviceSynchronize();

        update_particles<<<blocks, threads_per_block>>>(s.nbpart, s.d_mass, s.d_x, s.d_y, s.d_z,
                                                        s.d_vx, s.d_vy, s.d_vz, s.d_fx, s.d_fy, s.d_fz, dt);
        cudaDeviceSynchronize();

        if (step % printevery == 0) {
            s.copy_to_host();
            dump_state(s);
        }
    }

    s.free_device_memory();

    return 0;
}
