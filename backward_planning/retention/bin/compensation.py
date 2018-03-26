#!/usr/bin/env python3
import os

class Compensator(object):
    """Tools for compensating MTurk workers."""
    def __init__(self, use_sandbox=False, stdout_log=False, verbose=1,
                 verify_mturk_ssl=True, aws_key=None, aws_secret_key=None):

        self.verbose = verbose
        if aws_key is None:
            try:
                aws_key = os.environ['AWS_ACCESS_KEY_ID']
            except ValueError:
                raise ValueError('You must specify aws_key or '
                                 'have an environment varibale AWS_ACCESS_KEY_ID')
        if aws_secret_key is None:
            try:
                aws_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
            except ValueError:
                raise ValueError('You must specify aws_secret_key or '
                                 'have an environment varibale AWS_ACCESS_KEY_ID')

        self.conn = MechanicalTurk({
            'use_sandbox': use_sandbox,
            'stdout_log': stdout_log,
            'verify_mturk_ssl': verify_mturk_ssl,
            'aws_key': aws_key,
            'aws_secret_key': aws_secret_key,
        })
            
    def request(self, name, **params):
        """Make a request to the AWS API.

        http://docs.aws.amazon.com/AWSMechTurk/latest/AWSMturkAPI/ApiReference_OperationsArticle.html
        """
        return self.conn.request(name, params)

    def get_status(self, assignment_id):
        """Assignment status, one of Accepted, Rejected, Submitted"""
        r = self.request('GetAssignment',
            AssignmentId=assignment_id,
           )
        status = r.lookup('AssignmentStatus')
        if status:
            return status
        return r
        
    def get_bonus(self, assignment_id):
        """Ammount the worker has already been bonused."""
        r = self.request('GetBonusPayments',
                         AssignmentId=assignment_id)
        if int(r.lookup('NumResults')) == 0:
            return None
        return float(r.lookup('Amount'))

    def approve(self, assignment_id):
        """Approve an assignment.

        Returns response only if there is an error."""
        r = self.request('ApproveAssignment',
                         AssignmentId=assignment_id)
        if r.valid:
            self._log(2, 'Approved assignment {}'.format(assignment_id))
        elif r.lookup('CurrentState') == 'Approved':
            self._log(2, 'Already approved {}'.format(assignment_id))
        else:
            self._log(1, 'Error approving {}'.format(assignment_id))
            return r

    def grant_bonus(self, worker_id, assignment_id, bonus, reason='Performance bonus', repeat=False):
        """Grant a bonus for an assignment.

        If repeat is False, no action will be taken if assignment has 
        already been bonused. Returns response only if there if an error.
        """
        if not repeat:
            if self.get_bonus(assignment_id):
                self._log(2, 'Skipping previously bonused worker {}'.format(worker_id))
                return
        r = self.request('GrantBonus',
                         WorkerId=worker_id,
                         AssignmentId=assignment_id,
                         BonusAmount={'Amount': bonus, 'CurrencyCode': 'USD'},
                         Reason=reason)
        if r.valid:
            self._log(2, 'Bonused ${} to worker {}'.format(bonus, worker_id))
        else:
            self._log(1, 'Error assigning bonus {} to worker {}, assignment {}'
                     .format(bonus, worker_id, assignment_id))
            self._log(2, r)
            return r

    def process_df(self, df):
        df.assignment_id.apply(self.approve)
        for i, row in df.iterrows():
            self.approve(row.assignment_id)
            self.grant_bonus(row.worker_id, row.assignment_id, row.bonus)

    def _log(self, verb, *msg):
        if self.verbose >= verb:
            print('Compensator:', *msg)

# ============================================= #
# ========= Mechanical Turk interface ========= #
# ============================================= #

# This is a slightly modified version of 
#   https://github.com/ctrlcctrlv/mturk-python
# I include it here so that this single file can be distributed.

import time
import hmac
import hashlib
import base64
import json
import requests
import logging
import xmltodict
import collections
import six
from six.moves import range


class MechanicalTurk(object):
    def __init__(self, config_dict=None, config_file='mturkconfig.json'):
        """
        Use mturk_config_file to set config dictionary.
        Update the config dictionary with values from mturk_config_dict (if present).
        """
        try:
            mturk_config_dict = json.load(open(config_file))
        except IOError:
            mturk_config_dict = {}
        if config_dict is not None:
            mturk_config_dict.update(config_dict)
        if not mturk_config_dict.get("stdout_log"):
            logging.getLogger('requests').setLevel(logging.WARNING)

        self.sandbox = mturk_config_dict.get("use_sandbox") == True # Use sandbox?
        self.verify_mturk_ssl = mturk_config_dict.get("verify_mturk_ssl") == True
        self.aws_key = mturk_config_dict["aws_key"]
        self.aws_secret_key = str(mturk_config_dict["aws_secret_key"])
        self.request_retry_timeout = mturk_config_dict.get("request_retry_timeout") or 10
        self.max_request_retries = mturk_config_dict.get("max_request_retries") or 5

    def _generate_timestamp(self, gmtime):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", gmtime)

    def _generate_signature_python2(self, operation, timestamp, secret_access_key):
        my_sha_hmac = hmac.new(secret_access_key, 'AWSMechanicalTurkRequester' + operation + timestamp, hashlib.sha1)
        my_b64_hmac_digest = base64.encodestring(my_sha_hmac.digest()).strip()
        return my_b64_hmac_digest

    def _generate_signature_python3(self, operation, timestamp, secret_access_key):
        key = bytearray(secret_access_key, 'utf-8')
        msg = bytearray('AWSMechanicalTurkRequester' + operation + timestamp, 'utf-8')
        my_sha_hmac = hmac.new(key, msg, hashlib.sha1)
        my_b64_hmac_digest = base64.encodebytes(my_sha_hmac.digest()).strip()
        return str(my_b64_hmac_digest, 'utf-8')

    def _generate_signature(self, operation, timestamp, secret_access_key):
        if six.PY2:
            return self._generate_signature_python2(operation, timestamp, secret_access_key)
        elif six.PY3:
            return self._generate_signature_python3(operation, timestamp, secret_access_key)


    def _flatten(self, obj, inner=False):
        if isinstance(obj, collections.Mapping):
            if inner: obj.update({'':''})
            iterable = obj.items()
        elif isinstance(obj, collections.Iterable) and not isinstance(obj, six.string_types):
            iterable = enumerate(obj, start=1)
        else:
            return {"": obj}

        rv = {}
        for key, value in iterable:
            for inner_key, inner_value in self._flatten(value, inner=True).items():
                if inner_value != '':
                    rv.update({("{}.{}" if inner_key else "{}{}").format(key, inner_key): inner_value})
        return rv

    def request(self, operation, request_parameters={}):
        """Create a Mechanical Turk client request. Unlike other libraries
        (thankfully), my help ends here. You can pass the operation (view the
        list here: http://docs.amazonwebservices.com/AWSMechTurk/latest/AWSMtu
        rkAPI/ApiReference_OperationsArticle.html) as parameter one, and a
        dictionary of arguments as parameter two. To send multiple of the same
        argument (for instance, multiple workers to notify in NotifyWorkers),
        you can send a list."""
        
        if request_parameters is None:
            request_parameters = {}
        self.operation = operation

        if self.sandbox:
            self.service_url='https://mechanicalturk.sandbox.amazonaws.com/?Service=AWSMechanicalTurkRequester'
        else:
            self.service_url='https://mechanicalturk.amazonaws.com/?Service=AWSMechanicalTurkRequester'
        # create the operation signature
        timestamp = self._generate_timestamp(time.gmtime())
        signature = self._generate_signature(operation, timestamp, self.aws_secret_key)

        # Add common parameters to request dict
        request_parameters.update({"Operation":operation,"Version":"2014-08-15","AWSAccessKeyId":self.aws_key,"Signature":signature,"Timestamp":timestamp})

        self.flattened_parameters = self._flatten(request_parameters)

        # Retry request in case of ConnectionError
        req_retry_timeout = self.request_retry_timeout
        for i in range(self.max_request_retries):
            try:
                request = requests.post(self.service_url, params=self.flattened_parameters, verify=self.verify_mturk_ssl)
            except requests.exceptions.ConnectionError as e:
                last_requests_exception = e
                time.sleep(req_retry_timeout)
                req_retry_timeout *= 2
                continue
            break
        else:
            raise last_requests_exception

        request.encoding = 'utf-8'
        xml = request.text # Store XML response, might need it
        response = xmltodict.parse(xml.encode('utf-8'), dict_constructor=dict)
        return MechanicalTurkResponse(response, xml=xml)
    
    def external_form_action(self):
        """Return URL to use in the External question and HTML question form submit action."""
        if self.sandbox:
            return 'https://workersandbox.mturk.com/mturk/externalSubmit'
        else:
            return 'https://www.mturk.com/mturk/externalSubmit'

class MechanicalTurkResponse(dict):
    def __init__(self, response, xml=None):
        dict.__init__(self, response)
        self.response = response
        self.xml = xml
        req = self.lookup("Request")
        self.valid = req.get("IsValid") == "True" if req else False

    # @staticmethod
    # def shorten(r):
    #     assert len(r) == 1
    #     r1 = r[name + 'Response']
    #     r1.pop('OperationRequest')
    #     assert len(r1) == 1
    #     return r1.popitem()[1]

    def _find_item(self, obj, key):
        if isinstance(obj, list):
            for x in obj:
                yield from self._find_item(x, key)
        elif isinstance(obj, dict):
            if set(obj.keys()) == {'Key', 'Value'}:
                if obj['Key'] == key:
                    yield obj['Value']
            if key in obj:
                yield obj[key]

            for k, v in obj.items():
                yield from self._find_item(v, key)


    def lookup(self, element):
        return next(self._find_item(self.response, element), None)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import pandas as pd

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='Approves and assigns bonuses.'
    )
    parser.add_argument(
        'version',
        nargs='+',
        help='Version code e.g. 1A.0'
    )
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        default=0,
        )
    parser.add_argument(
        '--extra',
        type=float,
        default=0.0,
        )
    parser.add_argument(
        '-y',
        action='store_true'
        )
    parser.add_argument(
        '-f',
        action='store_true'
        )
    args = parser.parse_args()
    comp = Compensator(verbose=args.verbosity)

    for version in args.version:
        version = version.split('data/human_raw/')[-1]
        print(f'Approving and bonusing participants for {version}')
        try:
            identifiers = pd.read_csv(f'data/human_raw/{version}/identifiers.csv').set_index('pid')
            pdf = pd.read_csv(f'data/human/{version}/participants.csv').set_index('pid')
            pdf = pdf.join(identifiers).reset_index()

            try:
                payment = pd.read_csv(f'data/human_raw/{version}/payment.csv').set_index('worker_id')
                pdf = pdf.set_index('worker_id')
                pdf['status'] = payment['status'].fillna('submitted')
                pdf = pdf.reset_index()
            except FileNotFoundError:
                pdf['status'] = 'submitted'

            pdf.bonus = pdf.bonus.clip(lower=0) + args.extra
            total = pdf.query('status != "bonused"').bonus.sum()
            if not args.y:
                response = input(f'Assigning ${total:.2f} in bonuses. Continue? y/[n]:  ')
                if response != 'y':
                    print('Exiting.')
                    exit(0)

            def run():
                for i, row in pdf.iterrows():
                    if row.status == 'submitted':
                        err = comp.approve(row.assignment_id)
                        if not err:
                            row.status = 'approved'
                    if row.status == 'approved':
                        if row.bonus <= 0:
                            row.status = 'no bonus'
                        else:
                            err = comp.grant_bonus(row.worker_id, 
                                                   row.assignment_id,
                                                   round(row.bonus, 2))
                            if not err:
                                row.status = 'bonused'
                    yield row[['pid', 'worker_id', 'assignment_id', 'bonus', 'status']]

            pd.DataFrame(run()).to_csv(f'data/human_raw/{version}/payment.csv', index=False)

        except Exception as e:
            if args.f:
                print(e)
            else:
                raise

