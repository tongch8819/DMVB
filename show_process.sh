pid=$(ps aux | grep cartpole | tail -n1 | awk '{print $2}')
echo "PID: $pid"
ps -p $pid -o pid,comm,etime,time,cputime