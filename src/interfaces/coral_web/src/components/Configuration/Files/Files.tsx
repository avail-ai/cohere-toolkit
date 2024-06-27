import { Fragment, useEffect, useMemo, useState } from 'react';

import { ListFile } from '@/cohere-client';
import { BasicButton, Checkbox, Icon, Text, Tooltip } from '@/components/Shared';
import { useFocusFileInput } from '@/hooks/actions';
import { useFilesInConversation } from '@/hooks/files';
import { useConversationStore, useParamsStore } from '@/stores';
import { cn, formatFileSize } from '@/utils';

export interface UploadedFile extends ListFile {
  checked: boolean;
}

/**
 * Contains the file uploader and a list of all files being uploaded + that have been uploaded
 */
const Files: React.FC = () => {
  const {
    params: { fileIds },
    setParams
  } = useParamsStore();
  const { isFileInputQueuedToFocus, focusFileInput } = useFocusFileInput();
  const { removeFile } = useFilesInConversation();
  const {
    conversation: { id: conversationId, files }
  } = useConversationStore();

  useEffect(() => {
    if (isFileInputQueuedToFocus) {
      focusFileInput();
    }
  }, [isFileInputQueuedToFocus]);

  const uploadedFiles: UploadedFile[] = useMemo(() => {
    if (!files) return [];

    return files
      .map((document: ListFile) => ({
        ...document,
        checked: (fileIds || []).some((id) => id === document.id),
      }))
      .sort(
        (a, b) => new Date(b.created_at || '').getTime() - new Date(a.created_at || '').getTime()
      );
  }, [files, fileIds]);


  return (
    <div className="flex w-full flex-col gap-y-6">
      {uploadedFiles.length > 0 && (
        <div className="flex flex-col gap-y-14 pb-2">
          <div className="flex flex-col gap-y-4">
            {
              conversationId && uploadedFiles.length > 0 && (
                <Fragment key={'Files'}>
                  {uploadedFiles.map(
                    (f) => (
                      <FileControl
                        key={f.id}
                        file={f}
                        onDelete={() => removeFile(f.id, conversationId)}
                        onChange={(file, selected) => {
                          file.checked = selected;
                          if (selected) {
                            setParams({ fileIds: [...(fileIds || []), file.id] })
                          }
                          else {
                            setParams({ fileIds: (fileIds || []).filter((id) => id !== file.id) });
                          }
                        }} />
                    )
                  )}
                </Fragment>
              )
            }
          </div>
        </div>
      )}
    </div>
  );
};

type FileControlProps = {
  file: UploadedFile;
  onDelete: (file: UploadedFile) => void;
  onChange: (file: UploadedFile, selected: boolean) => void;
};

/**
 * @description Renders the entire conversation pane, which includes the header, messages,
 * composer, and the citation panel.
 */
const FileControl: React.FC<FileControlProps> = ({
  file,
  onDelete,
  onChange
}) => {

  const [hover, setHover] = useState(false);

  return (
    <div className=" group flex w-full flex-col gap-y-2 cursor-pointer"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={() => onChange(file, !file.checked)}>
      <div className="flex w-full items-center justify-between gap-x-2"
        onClick={(e) => {
          // make sure checkbox is not clicked
          e.preventDefault(); return false;
        }}>
        <Checkbox
          checked={file.checked}
          onChange={(e) => {
            e.preventDefault(); return false;
          }}
          theme="secondary"
          labelClassName={cn({
            'text-volcanic-500': false,
          })}
          disabled={false}
        />
        <div className={cn('flex w-[60%] lg:w-[70%]')}>
          <Text className="ml-0 w-full truncate">{file.file_name || ''}</Text>
        </div>
        <div className="flex h-5 w-32 grow items-center justify-end gap-x-1">
          {hover ? <BasicButton
            kind="minimal"
            size="sm"
            startIcon={<Icon name="close" />}
            onClick={() => onDelete(file)}
          /> : <Text styleAs="caption" className="text-volcanic-700">
            {formatFileSize(file.file_size ?? 0)}
          </Text>}
        </div>
      </div>
    </div>
  );
};

export const FilesSection: React.FC = () => {
  return (
    <section className="relative flex flex-col gap-y-8 px-5">
      <div className="flex gap-x-2">
        <Text styleAs="label" className="font-medium">
          Files in conversation
        </Text>
        <Tooltip label="To use uploaded files, at least 1 File Upload tool must be enabled" />
      </div>
      <Files />
    </section>
  );
};
