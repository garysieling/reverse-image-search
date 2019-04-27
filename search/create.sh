aws cloudformation delete-stack --stack-name es-test 
aws cloudformation create-stack --stack-name es-test --template-body file://./search.yaml --parameters ParameterKey=SearchEngineName,ParameterValue=test ParameterKey=ProductName,ParameterValue=test

