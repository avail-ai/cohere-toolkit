from abc import abstractmethod
from typing import Any, Dict, List, Optional

from fastapi import Request

from backend.database_models.database import DBSessionDep


class BaseTool:
    """
    Abstract base class for all Tools.

    Attributes:
        NAME (str): The name of the tool.
    """

    NAME = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._post_init_check()

    def _post_init_check(self):
        if any(
            [
                self.NAME is None,
            ]
        ):
            raise ValueError(f"{self.__name__} must have NAME attribute defined.")

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @classmethod
    @abstractmethod
    def _handle_tool_specific_errors(cls, error: Exception, **kwargs: Any) -> None:
        pass

    @abstractmethod
    async def call(self, parameters: dict, **kwargs: Any) -> List[Dict[str, Any]]: ...


class BaseToolAuthentication:
    """
    Abstract base class for Tool Authentication.
    """

    @classmethod
    @abstractmethod
    def get_auth_url(user_id: str) -> str: ...

    @classmethod
    @abstractmethod
    def is_auth_required(session: DBSessionDep, user_id: str) -> bool: ...

    @classmethod
    @abstractmethod
    def process_auth_token(request: Request, session: DBSessionDep) -> str: ...

    @classmethod
    @abstractmethod
    def get_token(user_id: str, session: DBSessionDep) -> Optional[str]:
        return None
