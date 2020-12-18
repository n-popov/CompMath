#include <iostream>
#include <fstream>
#include <unordered_map>
#include <boost/array.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint.hpp>


template<typename F, typename S>
auto middle(std::pair<F, S> bounds) {
    return (bounds.first + bounds.second) / 2.;
}


double y_init(double x) {
    return x * std::log(x);
}

using state_type = std::array<double, 3>;

void rhs(const state_type& R, state_type& dydx, const double x) {

    const auto& y = R[0], y_deriv = R[1], y_dderiv = R[2];

    dydx[0] = y_deriv;
    dydx[1] = y_dderiv;
    dydx[2] = x * std::exp(1) * y - std::pow(x, 3) * std::exp(1) * std::log(x) / 2 * y_deriv;
}

int main() {

    auto dopri5 = make_controlled(
            1e-9 , 1e-9 , boost::numeric::odeint::runge_kutta_dopri5<state_type>());

    const auto t = std::make_pair(std::exp(1), std::exp(2));
    constexpr const auto dt = 1e-6;

    std::pair bounds = {std::exp(1), 2 * std::exp(2)};
    std::pair deriv_bounds = {0, 5};
    state_type y;
    auto delta = 1e-4;
    y[0] = 10 * delta;
    std::unordered_map<double, state_type> buffer;

    for(; std::abs(y[0]) > delta;) {
        buffer.clear();
        y = {0., middle(deriv_bounds), 0.};
        integrate_adaptive(dopri5 , rhs , y, t.first , t.second , dt,
                           [&buffer](const state_type& x, const double t){buffer[t] = x;});
        if (y[0] < delta) {
            deriv_bounds.first = middle(deriv_bounds);
        } else if (y[0] > delta) {
            deriv_bounds.second = middle(deriv_bounds);
        }
    }

    std::ofstream out;
    out.open( "output5.txt" );
    out.precision(3);

    for(const auto& x : {0.5, 1., 1.5, 2., 2.5}) {
        std::cout << std::setw(3) << x << ' ' << y_init(x) + buffer[x][0] << '\n';
    }

    out.close();

}
