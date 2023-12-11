cd cpp
rm -rf build
mkdir build
cd build
cmake ..
make
mv mcts_tic_tac_toe.so ../../