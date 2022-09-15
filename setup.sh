conda create --file bsibo.yml

cat << EOF
conda activate bsibo
export PYTHONPATH="/home/v-tongcheng/Projects/BSIBO"
cd /home/v-tongcheng/Projects/BSIBO

RECOMMEND: modify your bashrc by the above lines.
EOF

