"""
Exception for prediction API.
"""

class APIError(Exception):
    # Adapted from code at http://flask.pocoo.org/docs/0.12/patterns/apierrors/
    status_code = 400

    def __init__(self, message, status_code=400, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

