mkdir -p logs

cd interval-bound-propagation

python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=1e-2,1e-3@15000,1e-4@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_mnist_01
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=5e-3,5e-4@15000,5e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_mnist_005
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-3,2e-4@15000,2e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_mnist_002
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=1e-3,1e-4@15000,1e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_mnist_001
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=5e-4,5e-5@15000,5e-6@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_mnist_0005
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-4,2e-5@15000,2e-6@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_mnist_0002

python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=1e-2,1e-3@15000,1e-4@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_mnist_ada_01
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=5e-3,5e-4@15000,5e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_mnist_ada_005
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-3,2e-4@15000,2e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_mnist_ada_002
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=1e-3,1e-4@15000,1e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_mnist_ada_001
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=5e-4,5e-5@15000,5e-6@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_mnist_ada_0005
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-4,2e-5@15000,2e-6@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_mnist_ada_0002

python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-3,2e-4@15000,2e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_mnist_ada_002_v2
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-3,2e-4@15000,2e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_mnist_ada_002_v3
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-3,2e-4@15000,2e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_mnist_ada_002_v4
python3 -u train.py --model=small --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-3,2e-4@15000,2e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_mnist_ada_002_v5


python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=1e-2,1e-3@60000,1e-4@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_cifar_01
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=5e-3,5e-4@60000,5e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_cifar_005
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=2e-3,2e-4@60000,2e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_cifar_002
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=1e-3,1e-4@60000,1e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_cifar_001
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=5e-4,5e-5@60000,5e-6@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_cifar_0005
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=2e-4,2e-5@60000,2e-6@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_cifar_0002

python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=1e-2,1e-3@60000,1e-4@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_cifar_ada_01
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=5e-3,5e-4@60000,5e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_cifar_ada_005
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=2e-3,2e-4@60000,2e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_cifar_ada_002
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=1e-3,1e-4@60000,1e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_cifar_ada_001
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=5e-4,5e-5@60000,5e-6@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_cifar_ada_0005
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=2e-4,2e-5@60000,2e-6@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_cifar_ada_0002

python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=5e-4,5e-5@60000,5e-6@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_cifar_ada_0005_v2
python3 -u train.py --model=small --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=5e-4,5e-5@60000,5e-6@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_cifar_ada_0005_v3


python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=1e-2,1e-3@15000,1e-4@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_medium_mnist_01
python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=5e-3,5e-4@15000,5e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_medium_mnist_005
python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-3,2e-4@15000,2e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_medium_mnist_002
python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=1e-3,1e-4@15000,1e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_medium_mnist_001
python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=5e-4,5e-5@15000,5e-6@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_medium_mnist_0005
python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-4,2e-5@15000,2e-6@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_medium_mnist_0002

python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=1e-2,1e-3@15000,1e-4@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_medium_mnist_ada_01
python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=5e-3,5e-4@15000,5e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_medium_mnist_ada_005
python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-3,2e-4@15000,2e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_medium_mnist_ada_002
python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=1e-3,1e-4@15000,1e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_medium_mnist_ada_001
python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=5e-4,5e-5@15000,5e-6@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_medium_mnist_ada_0005
python3 -u train.py --model=medium --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=2e-4,2e-5@15000,2e-6@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000   --test_every_n=600 --reg=ada --output_dir=ibp_medium_mnist_ada_0002


python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=1e-2,1e-3@60000,1e-4@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_large_cifar_01
python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=5e-3,5e-4@60000,5e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_large_cifar_005
python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=2e-3,2e-4@60000,2e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_large_cifar_002
python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=1e-3,1e-4@60000,1e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_large_cifar_001
python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=5e-4,5e-5@60000,5e-6@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_large_cifar_0005
python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=2e-4,2e-5@60000,2e-6@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --output_dir=ibp_large_cifar_0002

python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=1e-2,1e-3@60000,1e-4@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_large_cifar_ada_01
python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=5e-3,5e-4@60000,5e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_large_cifar_ada_005
python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=2e-3,2e-4@60000,2e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_large_cifar_ada_002
python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=1e-3,1e-4@60000,1e-5@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_large_cifar_ada_001
python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=5e-4,5e-5@60000,5e-6@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_large_cifar_ada_0005
python3 -u train.py --model=large --dataset=cifar10 --batch_size=50 --epsilon=0.03137 --epsilon_train=0.03451 --learning_rate=2e-4,2e-5@60000,2e-6@90000 --steps=350000 --warmup_steps=5000 --rampup_steps=50000 --test_every_n=1000 --reg=ada --output_dir=ibp_large_cifar_ada_0002


python3 -u train.py --model=wide --dataset=mnist --batch_size=100 --epsilon=0.3 --epsilon_train=0.3 --learning_rate=1e-3,1e-4@15000,1e-5@25000 --steps=60000 --warmup_steps=2000 --rampup_steps=10000 --test_every_n=600 --output_dir=ibp_wide_mnist_001


cd ..
mv interval-bound-propagation/networks/* networks/