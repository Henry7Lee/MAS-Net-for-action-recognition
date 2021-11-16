#v1_8
python main.py --lr 0.01 --epochs 70 --dataset_path ../../datasets/ --dataset somethingv1 \
                --arch resnet50 --num_segments 8 --store_name MAS_Net[v1_8] \
                --gpus 0 1 2 3 --workers 8 --batch-size 32 \
                --dropout 0.5 --weight_decay 5e-4 \
                --lr_type MAS_lr --lr_steps 30 40 50 60

#v2_16
python main.py --lr 0.005 --epochs 70 --dataset_path ../../datasets/ --dataset somethingv2 \
                --arch resnet50 --num_segments 16 --store_name MAS_Net[v2_16] \
                --gpus 0 1 2 3 --workers 8 --batch-size 16 \
                --dropout 0.8 --weight_decay 5e-4\
                --lr_type MAS_lr --lr_steps 30 40 50 60

#k400_8
python main.py --lr 0.01 --epochs 100 --dataset_path ../../datasets/ --dataset kinetics \
                --arch resnet50 --num_segments 8 --store_name MAS_Net[k400_8] \
                --gpus 0 1 2 3 --workers 32 --batch-size 32 \
                --dropout 0.5 --weight_decay 1e-4 \
                --lr_type step_lr --lr_steps 65 75 90