# this file imports custom routes into the experiment server

from flask import Blueprint, Response, abort, current_app, request, jsonify
from traceback import format_exc

from psiturk.psiturk_config import PsiturkConfig
from psiturk.user_utils import PsiTurkAuthorization, nocache
from psiturk.experiment_errors import ExperimentError

import json

# Database setup
from psiturk.db import db_session
from psiturk.models import Participant

# load the configuration options
config = PsiturkConfig()
config.load_config()

# if you want to add a password protect route use this
myauth = PsiTurkAuthorization(config)

# explore the Blueprint
custom_code = Blueprint(
    'custom_code', __name__,
    template_folder='templates',
    static_folder='static')

# Status codes
NOT_ACCEPTED = 0
ALLOCATED = 1
STARTED = 2
COMPLETED = 3
SUBMITTED = 4
CREDITED = 5
QUITEARLY = 6
BONUSED = 7
BAD = 8


def get_participants(codeversion):
    participants = Participant\
        .query\
        .filter(Participant.codeversion == codeversion)\
        .filter(Participant.status > 2)\
        .all()
    return participants


@custom_code.route('/data/<codeversion>/<name>', methods=['GET'])
@myauth.requires_auth
@nocache
def download_datafiles(codeversion, name):
    contents = {
        "trialdata": lambda p: p.get_trial_data(),
        "eventdata": lambda p: p.get_event_data(),
        "questiondata": lambda p: p.get_question_data()
    }

    if name not in contents:
        abort(404)

    query = get_participants(codeversion)
    data = []
    for p in query:
        try:
            data.append(contents[name](p))
        except TypeError:
            current_app.logger.error("Error loading {} for {}".format(name, p))
            current_app.logger.error(format_exc())
    ret = "".join(data)
    response = Response(
        ret,
        content_type="text/csv",
        headers={
            'Content-Disposition': 'attachment;filename=%s.csv' % name
        })

    return response


MAX_BONUS = 10

@custom_code.route('/compute_bonus', methods=['GET'])
def compute_bonus():
    # check that user provided the correct keys
    # errors will not be that gracefull here if being
    # accessed by the Javascrip client
    if not request.args.has_key('uniqueId'):
        raise ExperimentError('improper_inputs')

    # lookup user in database
    uniqueid = request.args['uniqueId']
    user = Participant.query.\
           filter(Participant.uniqueid == uniqueid).\
           one()

    final_bonus = 'NONE'
    # load the bonus information
    try:
        all_data = json.loads(user.datastring)
        question_data = all_data['questiondata']
        final_bonus = question_data['final_bonus']
        final_bonus = round(float(final_bonus), 2)
        if final_bonus > MAX_BONUS:
            raise ValueError('Bonus of {} excedes MAX_BONUS of {}'
                             .format(final_bonus, MAX_BONUS))
        user.bonus = final_bonus
        db_session.add(user)
        db_session.commit()

        resp = {
            'uniqueId': uniqueid,
            'bonusComputed': 'success',
            'bonusAmount': final_bonus
        }

    except:
        current_app.logger.error('error processing bonus for {}'.format(uniqueid))
        current_app.logger.error(format_exc())
        resp = {
            'uniqueId': uniqueid,
            'bonusComputed': 'failure',
            'bonusAmount': final_bonus
        }

    current_app.logger.info(str(resp))
    return jsonify(**resp)
