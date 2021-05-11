"""
Custom BPSTDP Learning Rule
"""
from bindsnet.learning import LearningRule

from abc import ABC
from typing import Union, Optional, Sequence

import torch
import numpy as np

from bindsnet.network.topology import (
    AbstractConnection,
    Connection,
    LocalConnection,
)
# from ..utils import im2col_indices

class BPSTDP(LearningRule):
    # language=rst
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. By default,
    pre-synaptic update is negative and the post-synaptic update is positive.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.
        :param connection: An ``AbstractConnection`` object whose weights the
            ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        # Ignores the ConvConnection
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        # Pre-synaptic update.
        if self.nu[0]:
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]
            self.connection.w -= self.reduction(torch.bmm(source_s, target_x), dim=0)
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1]:
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            )
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            self.connection.w += self.reduction(torch.bmm(source_x, target_s), dim=0)
            del source_x, target_s

        super().update()