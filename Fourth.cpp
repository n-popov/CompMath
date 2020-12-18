#include <iostream>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <boost/array.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/adams_bashforth_moulton.hpp>
#include <boost/numeric/odeint.hpp>


template<typename F, typename S>
auto middle(std::pair<F, S> bounds) {
    return (bounds.first + bounds.second) / 2.;
}

constexpr static auto pi = 3.141'592'653'589'793'2;


using state_type = std::array<double, 3>;

void rhs(const state_type& R, state_type& dydx, const double x) {

    const auto& y = R[0], y_deriv = R[1], y_dderiv = R[2];

    dydx[0] = y_deriv;
    dydx[1] = y_dderiv;
    dydx[2] = 2 - 6 * x + 2 * std::pow(x, 3) + (x * x - 3) * std::exp(x) * std::sin(x) * (1 + std::cos(x))
            + cos(x) * (std::exp(x) + (x * x - 1) + std::pow(x, 4) - 3 * x * x)
            - (x * x - 3) * std::cos(x) * y
            - (x * x - 3) * y_deriv;
}

int main() {

    constexpr const auto t = std::make_pair(0., pi);
    constexpr const auto dt = 1e-4;

    std::pair bounds = {0, pi * pi};
    std::pair deriv_bounds = {1.1, 1.2};
    state_type y;
    y[0] = 0;
    auto delta = 1e-4;

    std::unordered_map<double, state_type> buffer;

    for(; std::abs(y[0] - bounds.second) > delta;) {
        buffer.clear();
        y = {0, middle(deriv_bounds), 0.};
        integrate_const(boost::numeric::odeint::adams_bashforth_moulton<5, state_type>(),
                rhs , y, t.first , t.second , dt,
                [&buffer](const state_type& x, const double t){buffer[t] = x;});
        if (y[0] < bounds.second) {
            deriv_bounds.first = middle(deriv_bounds);
        } else if (y[0] > bounds.second) {
            deriv_bounds.second = middle(deriv_bounds);
        }
    }

//    std::ofstream out;
//    out.open( "output4.txt" );
//    out.precision(3);
//
//    for(const auto& [t, x] : buffer) {
//        out << t << ' ' << x[0] << std::endl;
//    }

    for(const auto& x : {0.5, 1., 1.5, 2., 2.5, 3.}) {
        std::cout << std::setw(3) << x << ' ' << buffer[x][0] << '\n';
    }

//    out.close();

}
