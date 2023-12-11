//
// Created by ignace konig on 10/12/2023.
//

#ifndef MCTS_TICTACTOE_INFERENCE_H
#define MCTS_TICTACTOE_INFERENCE_H

#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "game.cpp"
#include "net.cpp"
#include "mcts.cpp"


struct Sample {
    std::vector<float> input;
    std::vector<float> policy;
    float value;
    Sample(Eigen::VectorXf input, std::vector<float> policy, float value) : policy(std::move(policy)), value(value) {
        std::vector<float> inp(18, 0.0f);
        for (int i = 0; i < 18; i++) {
            inp[i] = input[i];
        }
        this->input = inp;
    }
};


class Inferencer {
public:
    Net net;

    Inferencer(Net net) : net(std::move(net)) {}

    std::vector<Sample> play_game() {
        State state = State();
        MCTS mcts = MCTS(state, net);
        std::vector<Sample> samples;
        std::vector<int> turns;
        while (!state.isTerminal()) {
            std::vector<float> policy = mcts.search(1000);
            // get random number between 0 and 1
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            float policy_sum = std::accumulate(policy.begin(), policy.end(), 0.0f);
            r *= policy_sum;
            float s = 0;
            int move;
            for (int i = 0; i < policy.size(); i++) {
                s += policy[i];
                if (r <= s) {
                    move = i;
                    break;
                }
            }
            samples.emplace_back(state.getNNInput(), policy, 0.0f);
            turns.push_back(state.turn);
            state.move(move);
            mcts.move_root(move);
        }
        float result = state.getWinner();
        for (int i = 0; i < samples.size(); i++) {
            samples[i].value = turns[i] * result;
        }
        return samples;
    }
    std::vector<Sample> get_samples(int n_samples) {
        std::vector<Sample> samples;
        while (samples.size() < n_samples) {
            std::vector<Sample> new_samples = play_game();
            samples.insert(samples.end(), new_samples.begin(), new_samples.end());
        }
        return samples;
    }
    std::vector<int> test_against_random(int n_games) {
        std::vector<int> results = {0, 0, 0};
        for (int i = 0; i < n_games; i++) {
            bool mcts_starts = i % 2 == 0;
            State state = State();
            MCTS mcts = MCTS(state, net);
            while (!state.isTerminal()) {
                int move;
                std::vector<float> policy = mcts.search(1000);
                if (mcts_starts && state.turn == 1 || !mcts_starts && state.turn == -1) {
                    // get random number between 0 and 1
                    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    float policy_sum = std::accumulate(policy.begin(), policy.end(), 0.0f);
                    r *= policy_sum;
                    float s = 0;
                    for (int i = 0; i < policy.size(); i++) {
                        s += policy[i];
                        if (r <= s) {
                            move = i;
                            break;
                        }
                    }
                } else {
                    std::vector<int> legal_moves = state.getLegalMoves();
                    int r = rand() % legal_moves.size();
                    move = legal_moves[r];
                }
                state.move(move);;
                mcts.move_root(move);
            }
            if (state.getWinner() == 0) {
                results[1]++;
            } else if ((state.getWinner() == 1 && mcts_starts) || (state.getWinner() == -1 && !mcts_starts)) {
                results[0]++;
            } else {
                results[2]++;
            }
        }
        return results;
    }
};


PYBIND11_MODULE(mcts_tic_tac_toe, m) {
    m.doc() = "pybind11 inference plugin"; // optional module docstring

    pybind11::class_<Sample>(m, "Sample")
            .def(pybind11::init<Eigen::VectorXf, std::vector<float>, float>())
            .def_readwrite("input", &Sample::input)
            .def_readwrite("policy", &Sample::policy)
            .def_readwrite("value", &Sample::value);

    pybind11::class_<Inferencer>(m, "Inferencer")
            .def(pybind11::init<Net>())
            .def("play_game", &Inferencer::play_game)
            .def("get_samples", &Inferencer::get_samples)
            .def("test_against_random", &Inferencer::test_against_random);

    pybind11::class_<Net>(m, "Net")
            .def(pybind11::init<std::vector<std::vector<float>>>());
}

#endif //MCTS_TICTACTOE_INFERENCE_H
