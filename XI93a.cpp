#include <iostream>
#include <array>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint.hpp>
#include <fstream>
#include <Python.h>

using state_type = std::array<double, 3>;

template<typename F, typename S>
auto middle(std::pair<F, S> bounds) {
    return (bounds.first + bounds.second) / 2.;
}

void rhs(const state_type& R, state_type& drdt, const double t) {

    const auto& y = R[0], y_deriv = R[1], y_dderiv = R[2];

    drdt[0] = y_deriv;
    drdt[1] = y_dderiv;
    drdt[2] = t * std::sqrt(y);
}

int main() {
    using stepper_t = boost::numeric::odeint::runge_kutta_dopri5<state_type>;
    auto stepper = make_controlled(1e-6, 1e-6, stepper_t());
    auto delta = 1e-4;
    std::pair bounds = {1., 2.};
    std::pair init_bounds = {1., 2.};
    std::pair x_bounds = {0., 1.};
    state_type currentR = {0, middle(init_bounds), 0};

    std::vector<std::pair<double, state_type>> buffer;

    for (; std::abs(bounds.second - currentR[0]) > delta;) {
        currentR = {0, middle(init_bounds), 0};
        buffer.clear();
        integrate_adaptive(stepper, rhs, currentR, x_bounds.first, x_bounds.second, 1e-6,
                           [&buffer](const state_type& x, const double t){buffer.emplace_back(t, x);});
        if (currentR[0] < bounds.second) {
            init_bounds.first = middle(init_bounds);
        } else if (currentR[0] > bounds.second) {
            init_bounds.second = middle(init_bounds);
        }
        std::cerr << currentR[0] << std::endl;
    }
    std::ofstream out;
    out.open( "outputXI93a.txt" );
    out.precision(3);

    for(const auto& [t, y] : buffer) {
        out << t << ' ' << y[0] << '\n';
    }
    out.close();

    char filename[] = "XI93a.py";
    FILE* fp;

    Py_Initialize();

    fp = _Py_fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);

    Py_Finalize();

}