#1-clip & center-crop & 224×224
python test.py --dataset somethingv1 --weights ./your.pth.tar --test_segments 8 --test_crops 1 --batch_size 32 -j 8

#2-clips & 3-crops & 256×256
python test.py --dataset somethingv1 --weights ./your.pth.tar --test_segments 8 --test_crops 3 --batch_size 8 -j 8 --full_res --twice_sample

#4-clips & 3-crops & 256×256
python test.py --dataset somethingv1 --weights ./your.pth.tar --test_segments 8 --test_crops 3 --batch_size 4 -j 8 --full_res --dense_sample --dense_sample_num 4

#10-clips & 3-crops & 256×256
python test.py --dataset somethingv1 --weights ./your.pth.tar --test_segments 8 --test_crops 3 --batch_size 2 -j 8 --full_res --dense_sample --dense_sample_num 10