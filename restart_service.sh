systemctl daemon-reload
systemctl stop perceptron.service
systemctl enable perceptron.service
systemctl start perceptron.service
systemctl status perceptron.service