#include <Python.h>

#include <iostream>
#include <fstream>
#include <boost/array.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/adams_bashforth_moulton.hpp>
#include <boost/numeric/odeint.hpp>

constexpr static auto pi = 3.141'592'653'589'793'2;

using state_type = std::array<double, 6>;

void rhs(const state_type& R, state_type& dRdt, const double t) {
    constexpr const auto gamma_m = 6.67 * 5.99e4;

    const auto x = R[0], y = R[1], x_deriv = R[2], y_deriv = R[3];

    const auto r = std::sqrt(x * x + y * y);

    dRdt[0] = x_deriv;
    dRdt[1] = y_deriv;
    dRdt[2] = - gamma_m * R[0] / std::pow(r, 3);
    dRdt[3] = - gamma_m * R[1] / std::pow(r, 3);
}

int main() {

    using abm = boost::numeric::odeint::adams_bashforth_moulton<5u, state_type>;

    constexpr auto r_c = 10000;

    constexpr auto v_c = 6.32;
    constexpr auto u = 0.74;
    constexpr std::pair t = {0, 10000};
    state_type R = {r_c, 0, 0, v_c - u};

    constexpr auto dt = 1;

    std::vector<std::pair<double, state_type>> buffer;

    boost::numeric::odeint::integrate_const(abm(), rhs , R, t.first , t.second , dt,
                           [&buffer](const state_type& x, const double t){buffer.emplace_back(t, x);});


    std::ofstream out;
    out.open( "output6.txt" );
    out.precision(3);

    double T;

    for(const auto& [t, y] : buffer) {
        out << y[0] << ' ' << y[1] << '\n';
        if (std::abs(y[0] - r_c) < 1) {
            T = t;
        }
    }

    std::cout << "T = " << T << ", from 3rd Kepler law:" <<
    2 * pi * std::pow((r_c + 6380) / 2, 1.5) / std::sqrt(6.67 * 5.99e4) << std::endl;

    out.close();

    char filename[] = "Sixth.py";
    FILE* fp;

    Py_Initialize();

    fp = _Py_fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);

    Py_Finalize();
    return 0;
}

