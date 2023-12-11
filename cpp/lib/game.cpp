//
// Created by ignace konig on 08/12/2023.
//

#include <vector>
#include <Eigen/Dense>

#ifndef MCTS_TICTACTOE_GAME_H
#define MCTS_TICTACTOE_GAME_H


class State {
public:
    std::vector<int> board;
    std::vector<std::vector<int>> lines = {{0, 1, 2},
                                          {3, 4, 5},
                                          {6, 7, 8},
                                          {0, 3, 6},
                                          {1, 4, 7},
                                          {2, 5, 8},
                                          {0, 4, 8},
                                          {2, 4, 6}};
    int turn;
    State() {
        board = std::vector<int>(9, 0);
        turn = 1;
    }
    void move(int pos) {
        board[pos] = turn;
        turn = -turn;
    }
    int getWinner() {
        for (auto line : lines) {
            int sum = 0;
            for (auto pos : line) {
                sum += board[pos];
            }
            if (sum == 3) {
                return 1;
            } else if (sum == -3) {
                return -1;
            }
        }
        return 0;
    }
    bool isTerminal() {
        if (getWinner() != 0) {
            return true;
        }
        return std::all_of(board.begin(), board.end(), [](int i) { return i != 0; });
    }
    Eigen::VectorXf getNNInput() {
        Eigen::VectorXf input(18);
        for (int i = 0; i < 9; i++) {
            input[i] = board[i] == turn ? 1.0f : 0.0f;
            input[i + 9] = board[i] == -turn ? 1.0f : 0.0f;
        }
        return input;
    }
    std::vector<int> getLegalMoves() {
        std::vector<int> legalMoves;
        for (int i = 0; i < 9; i++) {
            if (board[i] == 0) {
                legalMoves.push_back(i);
            }
        }
        return legalMoves;
    }
};


#endif //MCTS_TICTACTOE_GAME_H
