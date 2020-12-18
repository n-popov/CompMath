#include <iostream>
#include <fstream>
#include <boost/array.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint.hpp>
#include <Python.h>

using state_type = boost::array<double, 2>;

void rhs(const state_type& R, state_type& drdt, const double t) {

    constexpr auto mu = 1000;

    const auto& x = R[0], x_deriv = R[1];

    drdt[0] = x_deriv;
    drdt[1] = mu * (1 - x_deriv * x_deriv) * x_deriv + x;
}

int main() {

    auto dopri5 = make_dense_output(
            1e-4 , 1e-4 , boost::numeric::odeint::runge_kutta_dopri5<state_type>());

    state_type x = {0, 0.001};
    constexpr const auto t = std::make_pair(0., 1000.);
    constexpr const auto dt = 1e-4;

    std::ofstream out;
    out.open( "outputX93.txt" );
    out.precision(3);

    integrate_adaptive( dopri5 , rhs , x , t.first , t.second , dt , [&out](const state_type& x, const double t) {
        out << t << ' ' << x[0] << std::endl;
    } );

    out.close();

    char filename[] = "X93.py";
    FILE* fp;

    Py_Initialize();

    fp = _Py_fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);

    Py_Finalize();

}
