#include <iostream>
#include <fstream>
#include <boost/array.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint.hpp>
#include <Python.h>

using state_type = std::array<double, 4>;

constexpr static auto f = 0.1;

static inline auto compute_r1(double x, double mu, double y) {
    return std::sqrt(std::pow(x + mu, 2) + std::pow(y, 2));
}

static inline auto compute_r2(double x, double mu_line, double y) {
    return std::sqrt(std::pow(x - mu_line, 2) + std::pow(y, 2));
}

void rhs(const state_type& R, state_type& drdt, const double t) {

    constexpr auto mu = 0.012277471;
    constexpr auto mu_line = 1 - mu;

    const auto& x = R[0], y = R[1], x_deriv = R[2], y_deriv = R[3];

    drdt[0] = x_deriv;
    drdt[1] = y_deriv;

    auto r1 = compute_r1(x, mu, y);
    auto r2 = compute_r2(x, mu_line, y);

    drdt[2] = 2 * y_deriv + x - mu_line * (x + mu) / std::pow(r1, 3) - mu * (x - mu_line) / std::pow(r2, 3) - f * x_deriv;
    drdt[3] = -2 * x_deriv + y - mu_line * y / std::pow(r1, 3) - mu * y / std::pow(r2, 3) - f * y_deriv;

}

int main() {

    auto dopri5 = make_controlled(
            1e-9 , 1e-9 , boost::numeric::odeint::runge_kutta_dopri5<state_type>());

    state_type x = {0.994, 0, 0., -2.00158510637908252240537862224};
    constexpr const auto t = std::make_pair(0., 8.);
    constexpr const auto dt = 1e-9;

    std::ofstream out;
    out.open( "output7.txt" );
    out.precision(3);

    integrate_adaptive( dopri5 , rhs , x , t.first , t.second , dt , [&out](const state_type& x, const double t) {
        out << x[0] << ' ' << x[1] << std::endl;
        std::cout << t << std::endl;
    } );
    out.close();

    char filename[] = "Seventh.py";
    FILE* fp;

    Py_Initialize();

    fp = _Py_fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);

    Py_Finalize();
    return 0;

}
