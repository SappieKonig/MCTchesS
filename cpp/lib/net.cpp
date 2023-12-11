//
// Created by ignace konig on 09/12/2023.
//

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <tuple>

#ifndef MCTS_TICTACTOE_NET_H
#define MCTS_TICTACTOE_NET_H

auto relu = [](float x) { return std::max(0.0f, x); };

class Net {
private:
    Eigen::MatrixXf W1;
    Eigen::VectorXf b1;
    Eigen::MatrixXf Wp;
    Eigen::VectorXf bp;
    Eigen::MatrixXf Wv;
    Eigen::VectorXf bv;
public:
    Net() {
        W1 = Eigen::MatrixXf::Random(32, 18) * 0.01f;
        b1 = Eigen::VectorXf::Random(32) * 0.01f;
        Wp = Eigen::MatrixXf::Random(9, 32) * 0.01f;
        bp = Eigen::VectorXf::Random(9) * 0.01f;
        Wv = Eigen::MatrixXf::Random(1, 32) * 0.01f;
        bv = Eigen::VectorXf::Random(1) * 0.01f;
    }
    Net(std::vector<std::vector<float>> weights) {
        W1 = Eigen::Map<Eigen::Matrix<float, 32, 18, Eigen::RowMajor>>(weights[0].data());
        b1 = Eigen::Map<Eigen::VectorXf>(weights[1].data(), weights[1].size());
        Wp = Eigen::Map<Eigen::Matrix<float, 9, 32, Eigen::RowMajor>>(weights[2].data());
        bp = Eigen::Map<Eigen::VectorXf>(weights[3].data(), weights[3].size());
        Wv = Eigen::Map<Eigen::Matrix<float, 1, 32, Eigen::RowMajor>>(weights[4].data());
        bv = Eigen::Map<Eigen::VectorXf>(weights[5].data(), weights[5].size());
    }
    std::tuple<std::vector<float>, float> forward(const Eigen::VectorXf& input, const std::vector<int>& legalMoves) {
        Eigen::VectorXf h1 = W1 * input + b1;
        h1 = h1.unaryExpr(relu);

        Eigen::VectorXf p = Wp * h1 + bp;
        // set illegal moves to -inf
        for (int i = 0; i < 9; i++) {
            if (std::find(legalMoves.begin(), legalMoves.end(), i) == legalMoves.end()) {
                p[i] = -std::numeric_limits<float>::infinity();
            }
        }
        p = p.unaryExpr([](float x) { return std::exp(x); });
        p /= p.sum();
        std::vector<float> policy;
        policy.reserve(legalMoves.size());
        for (int legalMove : legalMoves) {
            policy.push_back(p[legalMove]);
        }
        std::vector<float> pVec(p.data(), p.data() + p.size());
        Eigen::VectorXf v = Wv * h1 + bv;
        v = v.unaryExpr([](float x) { return std::tanh(x); });
        float value = v[0];
        return {policy, value};
    }
    void update_net(std::vector<std::vector<float>> weights) {
        W1 = Eigen::Map<Eigen::Matrix<float, 32, 18, Eigen::RowMajor>>(weights[0].data());
        b1 = Eigen::Map<Eigen::VectorXf>(weights[1].data(), weights[1].size());
        Wp = Eigen::Map<Eigen::Matrix<float, 9, 32, Eigen::RowMajor>>(weights[2].data());
        bp = Eigen::Map<Eigen::VectorXf>(weights[3].data(), weights[3].size());
        Wv = Eigen::Map<Eigen::Matrix<float, 1, 32, Eigen::RowMajor>>(weights[4].data());
        bv = Eigen::Map<Eigen::VectorXf>(weights[5].data(), weights[5].size());
    }
};


//PYBIND11_MODULE(mcts_tic_tac_toe, m) {
//    pybind11::class_<Net>(m, "Net")
//            .def(pybind11::init<std::vector<std::vector<float>>>());
//}

#endif //MCTS_TICTACTOE_NET_H
