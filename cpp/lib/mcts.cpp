//
// Created by ignace konig on 09/12/2023.
//

#include <utility>

#include "game.cpp"
#include "net.cpp"

#ifndef MCTS_TICTACTOE_MCTS_H
#define MCTS_TICTACTOE_MCTS_H

class Node {
public:
    State state;
    std::vector<Node> children;
    std::vector<int> legalMoves;
    std::vector<float> policy;
    float value;
    int visits;
    Node(State state) {
        this->state = std::move(state);
        this->legalMoves = this->state.getLegalMoves();
        this->policy = std::vector<float>(this->legalMoves.size(), 0);
        this->value = 0.0f;
        this->visits = 0;
    }
    float score(int parent_visits, float parent_policy) {
        return (-value + 2.0f * parent_policy * (float) std::sqrt(parent_visits)) / ((float) visits + 1.0f);
    }
    Node* highest_scoring_child() {
        float best_score = -std::numeric_limits<float>::infinity();
        Node* best_child = nullptr;
        for (int i = 0; i < children.size(); i++) {
            float child_score = children[i].score(visits, policy[i]);
            if (child_score > best_score) {
                best_score = child_score;
                best_child = &children[i];
            }
        }
        return best_child;
    }
    void update(float result) {
        visits++;
        this->value += result;
    }
};



class MCTS {
public:
    Node root;
    Node* current;
    Net& net;
    MCTS(Net& net) : root(State()), net(net) { current = &root; }
    MCTS(State state, Net& net) : root(state), net(net) { current = &root; }
    std::vector<float> search(int iterations) {
        for (int i = 0; i < iterations; i++) {
            std::vector<Node*> path = select();
            float result = expand(path.back());
            backpropagate(path, result, path.back()->state.turn);
        }
        std::vector<float> policy(9, 0.0f);
        for (int i = 0; i < current->legalMoves.size(); i++) {
            policy[current->legalMoves[i]] = (float) current->children[i].visits / (float) current->visits;
        }
        return policy;
    }
    std::vector<Node*> select() {
        std::vector<Node*> path;
        Node* curr = current;
        path.push_back(curr);
        while (!curr->children.empty()) {
            curr = curr->highest_scoring_child();
            path.push_back(curr);
        }
        return path;
    }
    float expand(Node* node) {
        if (node->state.isTerminal()) {
            return (float) node->state.getWinner() * (float) node->state.turn;
        }
        auto input = node->state.getNNInput();
        auto [policy, value] = net.forward(input, node->legalMoves);
        node->policy = policy;
        for (int i = 0; i < node->legalMoves.size(); i++) {
            State child_state = node->state;
            child_state.move(node->legalMoves[i]);
            node->children.emplace_back(child_state);
        }
        return value;
    }
    void backpropagate(std::vector<Node*> path, float result, int result_turn) {
        for (auto node : path) {
            node->update(result * (float) node->state.turn * (float) result_turn);
        }
    }
    void move_root(int pos) {
        int move_index;
        for (int i = 0; i < current->legalMoves.size(); i++) {
            if (current->legalMoves[i] == pos) {
                move_index = i;
                break;
            }
        }
        current = &current->children[move_index];
    }
};


#endif //MCTS_TICTACTOE_MCTS_H
