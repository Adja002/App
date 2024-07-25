# aws_auth.py

import boto3
from botocore.exceptions import ClientError
import config

client = boto3.client('cognito-idp', region_name=config.AWS_REGION)

def sign_up(username, password, email):
    try:
        response = client.sign_up(
            ClientId=config.COGNITO_CLIENT_ID,
            Username=username,
            Password=password,
            UserAttributes=[
                {
                    'Name': 'email',
                    'Value': email
                },
            ]
        )
        return 'SignUp successful!'
    except ClientError as e:
        return e.response['Error']['Message']

def confirm_sign_up(username, confirmation_code):
    try:
        response = client.confirm_sign_up(
            ClientId=config.COGNITO_CLIENT_ID,
            Username=username,
            ConfirmationCode=confirmation_code
        )
        return 'Confirmation successful!'
    except ClientError as e:
        return e.response['Error']['Message']

def sign_in(username, password):
    try:
        response = client.initiate_auth(
            ClientId=config.COGNITO_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password
            }
        )
        return response['AuthenticationResult']
    except ClientError as e:
        return e.response['Error']['Message']
