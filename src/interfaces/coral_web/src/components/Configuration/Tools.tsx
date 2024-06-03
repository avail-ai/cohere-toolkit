import React from 'react';

import { FilesSection } from '@/components/Configuration/Files';
import { useFilesInConversation } from '@/hooks/files';
import { useConversationStore } from '@/stores';
import { cn } from '@/utils';

/**
 * @description Tools tab content that shows a list of available tools and files
 */
export const Tools: React.FC<{ className?: string }> = ({ className = '' }) => {
  const {
    conversation: { id: conversationId },
  } = useConversationStore();
  const { files } = useFilesInConversation();
  return (
    <article className={cn('flex flex-col pb-10', className)}>
      {/* File upload is not supported for conversarions without an id */}
      {conversationId && files.length > 0 && (
          <FilesSection />
      )}
    </article>
  );
};
