"""Websocket authentication.

The voice socket opened for anyone who could reach it, then built a pipeline
that reads and writes that person's memories and reminders.
"""

import pytest
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import User

from project.asgi import application

pytestmark = pytest.mark.django_db(transaction=True)


async def connect_as(user=None):
    communicator = WebsocketCommunicator(application, '/ws/voice/')
    communicator.scope['user'] = user or _anonymous()
    return communicator


def _anonymous():
    from django.contrib.auth.models import AnonymousUser

    return AnonymousUser()


async def test_anonymous_socket_is_refused():
    communicator = await connect_as()
    connected, _ = await communicator.connect()
    assert not connected, 'an unauthenticated socket must not be accepted'
    await communicator.disconnect()


async def test_missing_user_in_scope_is_refused():
    communicator = WebsocketCommunicator(application, '/ws/voice/')
    communicator.scope.pop('user', None)
    connected, _ = await communicator.connect()
    assert not connected
    await communicator.disconnect()
