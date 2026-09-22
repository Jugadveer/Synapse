"""Sign-up, sign-in and sign-out.

The original signup created the account without running any validator, stored
whatever came in as an age, and answered a duplicate username with a print
statement and a redirect.
"""

import pytest
from django.contrib.auth.models import User

from synapse.models import Profile

pytestmark = pytest.mark.django_db

GOOD_PASSWORD = 'a-good-password-42'


def signup(client, **overrides):
    form = {
        'signup_username': 'newperson',
        'signup_password': GOOD_PASSWORD,
        'name': 'New Person',
        'email': 'new@example.com',
        'gender': 'Female',
        'age': '72',
    }
    form.update(overrides)
    return client.post('/signup/', form)


# ----------------------------------------------------------------- signup

def test_a_valid_signup_creates_an_account_and_profile(client):
    response = signup(client)

    assert response.status_code == 302
    user = User.objects.get(username='newperson')
    profile = Profile.objects.get(user=user)
    assert profile.name == 'New Person'
    assert profile.age == 72
    assert user.check_password(GOOD_PASSWORD)


def test_signing_up_logs_the_person_in(client):
    signup(client)
    assert client.session.get('_auth_user_id')


@pytest.mark.parametrize('password', ['123', 'password', 'abc', '1234567'])
def test_a_weak_password_is_refused(client, password):
    """create_user does not run the configured validators by itself."""
    response = signup(client, signup_password=password)

    assert response.status_code == 200, 'should re-render with an error'
    assert not User.objects.filter(username='newperson').exists()


@pytest.mark.parametrize('age', ['not a number', '', 'twelve', '-5', '0', '200', '3.5'])
def test_a_bad_age_is_refused(client, age):
    """A non-numeric age used to reach the IntegerField and raise a 500."""
    response = signup(client, age=age)

    assert response.status_code == 200
    assert not User.objects.filter(username='newperson').exists()


def test_a_duplicate_username_is_refused(client):
    User.objects.create_user(username='newperson', password=GOOD_PASSWORD)

    response = signup(client)
    assert response.status_code == 200
    assert User.objects.filter(username='newperson').count() == 1


@pytest.mark.parametrize('field', ['signup_username', 'signup_password'])
def test_the_required_fields_are_required(client, field):
    response = signup(client, **{field: ''})

    assert response.status_code == 200
    assert not User.objects.filter(username='newperson').exists()


def test_a_username_of_spaces_is_refused(client):
    response = signup(client, signup_username='   ')
    assert response.status_code == 200
    assert User.objects.count() == 0


def test_no_half_made_account_is_left_behind(client):
    """The account and its profile are created together or not at all."""
    signup(client, age='nonsense')

    assert User.objects.filter(username='newperson').count() == 0
    assert Profile.objects.count() == 0


def test_signup_rejects_a_get(client):
    response = client.get('/signup/')
    assert response.status_code == 302
    assert User.objects.count() == 0


# ------------------------------------------------------------------ login

@pytest.fixture
def person():
    return User.objects.create_user(username='someone', password=GOOD_PASSWORD)


def test_correct_details_sign_in(client, person):
    response = client.post('/login/', {'username': 'someone', 'password': GOOD_PASSWORD})

    assert response.status_code == 302
    assert client.session.get('_auth_user_id') == str(person.pk)


def test_a_wrong_password_does_not_sign_in(client, person):
    response = client.post('/login/', {'username': 'someone', 'password': 'wrong-password'})

    assert response.status_code == 200, 'should re-render with an error'
    assert not client.session.get('_auth_user_id')


def test_an_unknown_username_does_not_sign_in(client):
    response = client.post('/login/', {'username': 'nobody', 'password': GOOD_PASSWORD})

    assert response.status_code == 200
    assert not client.session.get('_auth_user_id')


def test_empty_credentials_do_not_sign_in(client, person):
    response = client.post('/login/', {'username': '', 'password': ''})

    assert response.status_code == 200
    assert not client.session.get('_auth_user_id')


def test_login_rejects_a_get(client):
    assert client.get('/login/').status_code == 302


def test_the_failure_message_does_not_say_which_part_was_wrong(client, person):
    """Distinguishing them would confirm whether an account exists."""
    import re

    def message(response):
        # Compare the rendered error, not the whole page: the CSRF token
        # differs between any two responses.
        found = re.findall(rb'(Incorrect[^<]*|[^<>]*password[^<>]*)', response.content)
        return {part.strip() for part in found if part.strip()}

    wrong_password = message(
        client.post('/login/', {'username': 'someone', 'password': 'wrong-password'})
    )
    unknown_user = message(
        client.post('/login/', {'username': 'nobody', 'password': GOOD_PASSWORD})
    )
    assert wrong_password == unknown_user
    assert any(b'Incorrect' in part for part in wrong_password)


# ----------------------------------------------------------------- logout

def test_logging_out_clears_the_session(client, person):
    client.force_login(person)
    assert client.session.get('_auth_user_id')

    response = client.post('/logout/')
    assert response.status_code == 302
    assert not client.session.get('_auth_user_id')


def test_logout_rejects_a_get(client, person):
    """It used to fall off the end and return None, which Django turns into a 500."""
    client.force_login(person)
    assert client.get('/logout/').status_code == 405
    assert client.session.get('_auth_user_id'), 'still signed in'


# ------------------------------------------------------------- protection

@pytest.mark.parametrize('path', ['/dashboard/', '/dashboard-data/'])
def test_private_pages_need_a_session(client, path):
    response = client.get(path)
    assert response.status_code in (302, 401, 403)


def test_the_landing_page_is_public(client):
    assert client.get('/').status_code == 200


# ------------------------------------------------- errors reach the page

def test_a_failed_login_says_so_on_the_page(client, person):
    """The error was passed to a template that never rendered it.

    A person typing the wrong password saw the form reappear with no
    explanation at all.
    """
    response = client.post('/login/', {'username': 'someone', 'password': 'wrong-password'})
    assert b'Incorrect username or password' in response.content


@pytest.mark.parametrize('overrides,fragment', [
    ({'signup_password': '123'}, b'password'),
    ({'age': 'not a number'}, b'age as a number'),
    ({'signup_username': ''}, b'required'),
])
def test_a_failed_signup_says_why(client, overrides, fragment):
    response = signup(client, **overrides)
    assert fragment.lower() in response.content.lower()


def test_the_retry_form_posts_the_names_the_view_reads(client):
    """The stale templates offered `username`/`password` for signup.

    The view reads `signup_username`/`signup_password`, so every retry from
    that page failed with "Username and password are required".
    """
    response = signup(client, age='nonsense')

    assert b'name="signup_username"' in response.content
    assert b'name="signup_password"' in response.content


def test_signing_up_again_from_the_error_page_works(client):
    """The whole point: the second attempt has to be able to succeed."""
    signup(client, age='nonsense')
    assert not User.objects.filter(username='newperson').exists()

    assert signup(client).status_code == 302
    assert User.objects.filter(username='newperson').exists()
