python3 train.py --exp-name Evnondb \
--max-lr 1e-5 \
--train-bs 96 \
--val-bs 8 \
--weight-decay 0.5 \
--mask-ratio 0.4 \
--attn-mask-ratio 0.1 \
--max-span-length 8 \
--img-size 512 64 \
--proj 8 \
--dila-ero-max-kernel 2 \
--dila-ero-iter 1 \
--proba 0.5 \
--alpha 1 \
--total-iter 100000 \
--pretrained-model output/Evnondb/best_CER.pth \
Evnondb
